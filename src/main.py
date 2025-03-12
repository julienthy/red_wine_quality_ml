import logging

from exploratory_analysis import load_data, explore_dataset, split_features_target, dataset_summary
from data_preprocessing import preprocess_data
from model_training import train_and_evaluate
from pca_analysis import perform_pca

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb

import pandas as pd

# Configuration des logs pour un suivi professionnel
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(
    data_path: str = "./data/winequality-red.csv",
    output_dir: str = "./data/",
    target_col: str = "quality",
    test_size: float = 0.2,
    random_state: int = 42,
    binarize_threshold: int = None  # None pour multiclasse, ex. 6 pour binaire
) -> None:
    """
    Orchestre le pipeline de machine learning : chargement, exploration, prétraitement, PCA, entraînement.

    Args:
        data_path (str): Chemin vers le fichier de données.
        target_col (str): Nom de la colonne cible.
        test_size (float): Proportion des données pour le test.
        random_state (int): Graine aléatoire pour la reproductibilité.
        binarize_threshold (int, optional): Seuil pour binariser la cible (None pour multiclasse).
    """
    try:
        # Étape 1 : Chargement et exploration des données
        logging.info("Chargement des données...")
        df = load_data(data_path)
        if df is None:
            logging.error("Échec du chargement des données. Arrêt du programme.")
            return
        
        X, y = split_features_target(df, "quality")
        dataset_summary(X, y)
        
        logging.info("Exploration des données...")
        explore_dataset(X, y)

        # Étape 2 : Prétraitement des données
        logging.info("Prétraitement des données...")
        X_scaled, y, scaler = preprocess_data(
        file_path=data_path,
        output_dir=output_dir,
        target_column='quality',
        remove_duplicates_flag=False,  # Conserver les doublons
        outlier_method='none',    # Utiliser la winsorization
        winsorize_limits=(0.01, 0.99) # Limites à 1% et 99%
        )
        if X_scaled is None or y is None:
            logging.error("Échec du prétraitement. Arrêt du programme.")
            return

        # Étape 3 : Analyse PCA
        logging.info("Analyse PCA...")
        # Chargement des données prétraitées
        df = pd.read_csv("./data/df_preprocessed.csv")
        X_scaled, y = split_features_target(df, 'quality')

        # Extraction des noms des features (sans la cible 'quality')
        feature_names = df.drop('quality', axis=1).columns

        perform_pca(X_scaled, feature_names, n_components=5)

        # Étape 4 : Définition des modèles à entraîner
        models = [
            {
                'name': 'Logistic Regression',
                'model': LogisticRegression(max_iter=1000, random_state=random_state),
                'param_grid': {'C': [0.01, 0.1, 1, 10], 'solver': ['lbfgs', 'liblinear']}
            },
            {
                'name': 'Random Forest',
                'model': RandomForestClassifier(random_state=random_state),
                'param_grid': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
            },
            {
                'name': 'SVM',
                'model': SVC(random_state=random_state, probability=True),  # probability=True pour AUC-ROC
                'param_grid': {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto']
                }
            },
            {
                'name': 'KNN',
                'model': KNeighborsClassifier(),
                'param_grid': {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'p': [1, 2]  # Distance Manhattan (1) ou Euclidienne (2)
                }
            },
            {
                'name': 'Gradient Boosting',
                'model': GradientBoostingClassifier(random_state=random_state),
                'param_grid': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5]
                }
            },
            {
                'name': 'XGBoost',
                'model': xgb.XGBClassifier(eval_metric='mlogloss', random_state=random_state),
                'param_grid': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 1.0]
                }
            }
        ]

        # Étape 5 : Entraînement et évaluation
        logging.info(f"Entraînement et évaluation ({'binaire' if binarize_threshold else 'multiclasse'})...")
        results = train_and_evaluate(
            X=X_scaled,
            y=y,
            models=models,
            test_size=test_size,
            random_state=random_state,
            metric='f1_score',
            binarize_threshold=binarize_threshold,
            target_col=target_col,
            search_type='grid',
            n_iter=20
        )

        # Étape 6 : Résumé des résultats
        best_model_name = max(results, key=lambda k: results[k]['f1_score'])
        logging.info(f"Meilleur modèle : {best_model_name} avec F1-score = {results[best_model_name]['f1_score']:.3f}")

    except Exception as e:
        logging.error(f"Une erreur est survenue : {str(e)}")
        raise

if __name__ == "__main__":
    # Exécution pour multiclasse
    main()

    # Exécution pour binaire (qualité >= 6)
    # main(binarize_threshold=6)