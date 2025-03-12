from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from imblearn.over_sampling import SMOTE

import pandas as pd
import numpy as np

import logging
import joblib
from pathlib import Path
from typing import List, Dict, Union, Callable, Optional

from exploratory_analysis import split_features_target

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_and_evaluate_defaults(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    models: List[Dict[str, Union[str, Callable]]],
    binarize_threshold: Optional[int] = None
) -> tuple[Dict[str, Dict[str, Union[float, np.ndarray]]], Optional[Dict], Optional[Dict]]:
    """
    Entraîne et évalue plusieurs modèles avec leurs paramètres par défaut.

    Args:
        X (np.ndarray): Features prétraitées.
        y (pd.Series | pd.DataFrame): Variable cible (multiclasses ou multidimensionnelle).
        models (List[Dict]): Liste de dictionnaires avec 'name' et 'model'.
        binarize_threshold (int, optional): Seuil pour dichotomiser y. Si None, pas de binarisation.

    Returns:
        Dict[str, Dict[str, Union[float, np.ndarray]]]: Résultats pour chaque modèle.
    """
    # Binarisation optionnelle
    if binarize_threshold is not None:
        y_train = (y_train >= binarize_threshold).astype(int)
        y_test = (y_test >= binarize_threshold).astype(int)
        logging.info(f"Cible binarisée avec seuil >= {binarize_threshold}")
        label_map = None
        reverse_map = None
    else:
        original_labels = np.unique(y_train)
        label_map = {label: idx for idx, label in enumerate(sorted(original_labels))}
        reverse_map = {idx: label for label, idx in label_map.items()}
        y_train_mapped = np.array([label_map[val] for val in y_train])
        y_test_mapped = np.array([label_map[val] for val in y_test])
        y_train, y_test = y_train_mapped, y_test_mapped
    
    results = {}
    for model_dict in models:
        name = model_dict['name']
        model = model_dict['model']

        # Ajustement spécifique pour XGBoost en binaire
        if name == 'XGBoost' and binarize_threshold is not None:
            model.set_params(objective='binary:logistic')
        
        model.fit(X_train, y_train)
        y_pred_mapped = model.predict(X_test)

        if reverse_map:
            y_pred = np.array([reverse_map[pred] for pred in y_pred_mapped])
            y_test_original = np.array([reverse_map[val] for val in y_test])
        else:
            y_pred = y_pred_mapped
            y_test_original = y_test
        
        # Calcul des métriques
        metrics = {
            'accuracy': accuracy_score(y_test_original, y_pred),
            'f1_score': f1_score(y_test_original, y_pred, average='weighted'),
            'precision': precision_score(y_test_original, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test_original, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_test_original, y_pred).tolist()
        }
        
        # Ajout des probabilités pour AUC-ROC si binaire
        if binarize_threshold is not None and hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]  # Probabilité de la classe positive
            metrics['y_prob'] = y_prob
            metrics['y_test'] = y_test
        
        results[name] = metrics
        logging.info(f"{name} - Accuracy: {metrics['accuracy']:.3f}, F1-score: {metrics['f1_score']:.3f}")
    
    return results, label_map, reverse_map

def optimize_best_model(
    X: np.ndarray,
    y: Union[pd.Series, pd.DataFrame],
    model: Callable,
    param_grid: Dict,
    metric: str = 'accuracy',
    binarize_threshold: Optional[int] = None,
    cv_folds: int = 5,
    random_state: int = 42,
    search_type: str = 'grid',
    n_iter: int = 20
) -> Callable:
    """
    Optimise un modèle avec GridSearchCV et retourne le meilleur estimateur.

    Args:
        X (np.ndarray): Features.
        y (pd.Series | pd.DataFrame): Cible.
        model (Callable): Modèle à optimiser.
        param_grid (Dict): Grille d’hyperparamètres.
        metric (str): Métrique à optimiser ('accuracy', 'f1_weighted', etc.).
        cv_folds (int): Nombre de folds pour la validation croisée.
        random_state (int): Graine aléatoire.

    Returns:
        Callable: Meilleur modèle optimisé.
    """
    if binarize_threshold is not None:
        y = (y >= binarize_threshold).astype(int)
    else:
        # Remapping des étiquettes pour XGBoost en multiclasses
        original_labels = np.unique(y)
        label_map = {label: idx for idx, label in enumerate(sorted(original_labels))}
        y = np.array([label_map[val] for val in y])
        logging.info("Étiquettes remappées pour compatibilité avec XGBoost")

    # Ajustement de la métrique pour GridSearchCV
    scoring = metric
    if metric == 'f1_score':
        scoring = 'f1' if binarize_threshold is not None else 'f1_weighted'

    # Ajustement spécifique pour XGBoost en binaire
    if isinstance(model, xgb.XGBClassifier) and binarize_threshold is not None:
        model.set_params(objective='binary:logistic')

    if search_type == 'random':
        grid_search = RandomizedSearchCV(model, param_grid, n_iter=n_iter, cv=cv_folds, scoring=scoring, n_jobs=-1, random_state=random_state)
    else:
        grid_search = GridSearchCV(model, param_grid, cv=cv_folds, scoring=scoring, n_jobs=-1)

    # grid_search = GridSearchCV(model, param_grid, cv=cv_folds, scoring=scoring, n_jobs=-1)
    grid_search.fit(X, y)
    best_model = grid_search.best_estimator_
    logging.info(f"Meilleurs paramètres pour {model.__class__.__name__}: {grid_search.best_params_}")
    logging.info(f"Meilleur score ({metric}): {grid_search.best_score_:.3f}")
    return best_model

def balance_classes(X_train: np.ndarray, y_train: np.ndarray, method: str = 'smote', random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Équilibre les classes dans les données d'entraînement."""
    if method == 'smote':
        smote = SMOTE(random_state=random_state)
        X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
        logging.info("Classes équilibrées avec SMOTE")
    else:
        raise ValueError(f"Méthode de rééquilibrage inconnue : {method}")
    return X_train_bal, y_train_bal

def display_results(results: Dict[str, Dict[str, Union[float, np.ndarray]]]) -> None:
    """
    Affiche un rapport complet avec tableau des métriques et matrices de confusion.

    Args:
        results (Dict): Résultats des modèles issus de train_and_evaluate_defaults.
    """
    metrics_df = pd.DataFrame({
        name: {
            'Accuracy': metrics['accuracy'],
            'F1-score': metrics['f1_score'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall']
        } for name, metrics in results.items()
    }).T.round(3)
    
    print("\n=== Rapport des performances ===")
    print(metrics_df)
    
    print("\n=== Matrices de confusion ===")
    for name, metrics in results.items():
        print(f"\n{name}:")
        print(np.array(metrics['confusion_matrix']))

def display_auc_roc(results: Dict[str, Dict[str, Union[float, np.ndarray]]]) -> None:
    """
    Affiche l’AUC-ROC pour chaque modèle, uniquement pour une classification binaire.

    Args:
        results (Dict): Résultats des modèles issus de train_and_evaluate_defaults.
    """
    print("\n=== AUC-ROC (classification binaire uniquement) ===")
    for name, metrics in results.items():
        if 'y_prob' in metrics and 'y_test' in metrics:
            auc_roc = roc_auc_score(metrics['y_test'], metrics['y_prob'])
            print(f"{name}: AUC-ROC = {auc_roc:.3f}")
        else:
            print(f"{name}: AUC-ROC non calculable (classification multiclasses ou pas de probabilités)")

def cross_validate_model(model: Callable, X: np.ndarray, y: np.ndarray, cv: int = 5, scoring: str = 'f1_weighted') -> None:
    """Effectue une validation croisée sur le modèle et affiche les résultats."""
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    logging.info(f"Validation croisée ({scoring}) : Moyenne = {np.mean(scores):.3f}, Écart-type = {np.std(scores):.3f}")

def train_and_evaluate(
    X: np.ndarray,
    y: Union[pd.Series, pd.DataFrame],
    models: List[Dict[str, Union[str, Callable, Dict]]],
    test_size: float = 0.2,
    random_state: int = 42,
    metric: str = 'accuracy',
    binarize_threshold: Optional[int] = None,
    target_col: Optional[str] = None,
    balance_classes: bool = False,
    balance_method: str = 'smote',
    search_type: str = 'grid',
    n_iter: int = 20,
    output_dir: str = "./models"
) -> Dict[str, Dict[str, Union[float, Callable]]]:
    """
    Pipeline complet : évaluation par défaut, sélection et optimisation du meilleur modèle.

    Args:
        X (np.ndarray): Features.
        y (pd.Series | pd.DataFrame): Cible.
        models (List[Dict]): Liste avec 'name', 'model', 'param_grid'.
        test_size (float): Proportion du test set.
        random_state (int): Graine aléatoire.
        metric (str): Métrique pour sélectionner le meilleur modèle.
        binarize_threshold (int, optional): Seuil pour binariser y.
        target_col (str, optional): Colonne cible si y multidimensionnel.
        output_dir (str): Dossier pour sauvegarder le meilleur modèle.

    Returns:
        Dict[str, Dict[str, Union[float, Callable]]]: Résultats et meilleur modèle optimisé.
    """
    # On gère le cas où y serait multidimensionnel, puis on aplatit y en 1D
    if isinstance(y, pd.DataFrame):
        y = y[target_col].values.ravel() if target_col else y.iloc[:, 0].values.ravel()
    elif isinstance(y, pd.Series):
        y = y.values.ravel()

    # Séparation train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Rééquilibrage des classes
    if balance_classes:
        X_train, y_train = balance_classes(X_train, y_train, method=balance_method, random_state=random_state)

    # Étape 1 : Évaluation avec paramètres par défaut
    results, label_map, reverse_map = train_and_evaluate_defaults(X_train, y_train, X_test, y_test, models, binarize_threshold)
    display_results(results)
    
    # Affichage de l’AUC-ROC si binaire
    if binarize_threshold is not None:
        display_auc_roc(results)
    
    # Sélection du meilleur modèle selon la métrique choisie
    best_model_name = max(results, key=lambda k: results[k][metric])
    best_model = next(m for m in models if m['name'] == best_model_name)['model']
    param_grid = next(m for m in models if m['name'] == best_model_name).get('param_grid', {})
    
    logging.info(f"Meilleur modèle avec paramètres par défaut : {best_model_name} ({metric}: {results[best_model_name][metric]:.3f})")
    
    # Étape 2 : Optimisation du meilleur modèle
    if param_grid:
        optimized_model = optimize_best_model(X_train, y_train, best_model, param_grid, metric=metric, binarize_threshold=binarize_threshold, search_type=search_type, n_iter=n_iter, random_state=random_state)
        results[best_model_name]['optimized_model'] = optimized_model

        # Évaluation sur le test set
        y_pred_optimized = optimized_model.predict(X_test)

        if reverse_map:
            y_pred_optimized = np.array([reverse_map[pred] for pred in y_pred_optimized])
            y_test_eval = y_test
        else:
            y_test_eval = (y_test >= binarize_threshold).astype(int)

        f1_optimized = f1_score(y_test_eval, y_pred_optimized, average='weighted')
        logging.info(f"Score du modèle optimisé sur test set (F1-score) : {f1_optimized:.3f}")
        
        # Sauvegarde du modèle optimisé
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(optimized_model, output_path / f"best_model_{best_model_name.lower().replace(' ', '_')}.pkl")
        logging.info(f"Modèle optimisé sauvegardé dans {output_path / f'best_model_{best_model_name.lower().replace(' ', '_')}.pkl'}")
    
    return results

if __name__ == "__main__":
    # Chargement des données prétraitées
    df = pd.read_csv("./data/df_preprocessed.csv")
    X_scaled, y = split_features_target(df, 'quality')
    
    # Liste des modèles avec grilles d’hyperparamètres
    models = [
        {'name': 'Logistic Regression', 'model': LogisticRegression(max_iter=1000, random_state=42),
         'param_grid': {'C': [0.1, 1, 10], 'solver': ['lbfgs', 'liblinear']}},
        {'name': 'Random Forest', 'model': RandomForestClassifier(random_state=42),
         'param_grid': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}},
        {'name': 'SVM', 'model': SVC(random_state=42),
         'param_grid': {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']}},
        {'name': 'KNN', 'model': KNeighborsClassifier(),
         'param_grid': {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}},
        {'name': 'Gradient Boosting', 'model': GradientBoostingClassifier(random_state=42),
         'param_grid': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]}},
        {'name': 'XGBoost', 'model': xgb.XGBClassifier(eval_metric='mlogloss', random_state=42),
         'param_grid': {'n_estimators': [50, 100], 'max_depth': [3, 5], 'learning_rate': [0.01, 0.1]}}
    ]
    
    # Exécution multiclasses
    print("=== Classification multiclasses ===")
    results_multi = train_and_evaluate(X_scaled, y, models, metric='f1_score', target_col='quality')
    
    # Exécution binaire avec seuil
    print("\n=== Classification binaire (quality >= 6) ===")
    results_binary = train_and_evaluate(X_scaled, y, models, metric='f1_score', binarize_threshold=6, target_col='quality')