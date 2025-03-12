import pandas as pd
import numpy as np
from scipy.stats import mstats

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import IsolationForest

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

import logging
from pathlib import Path
from typing import Tuple, Union, List
import joblib


# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Charge un fichier de données avec détection automatique du séparateur et du type de fichier.
    
    Args:
        file_path (str/Path): Chemin vers le fichier de données
        
    Returns:
        pd.DataFrame: DataFrame contenant les données
        
    Raises:
        FileNotFoundError: Si le fichier n'existe pas
        ValueError: Si le format de fichier n'est pas supporté
    """

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Fichier {file_path} introuvable")
    
    suffix = file_path.suffix.lower()
    
    try:
        if suffix == ".csv":
            # Détection du délimiteur
            with open(file_path, 'r') as f:
                first_line = f.readline()
                delimiter = ';' if ';' in first_line else ','
            return pd.read_csv(file_path, delimiter=delimiter)
        
        elif suffix == ".xlsx":
            return pd.read_excel(file_path)
        
        elif suffix == ".json":
            return pd.read_json(file_path)
        
        else:
            raise ValueError(f"Format non supporté : {suffix}")
    
    except Exception as e:
        raise RuntimeError(f"Échec du chargement de {file_path} : {e}")


def split_features_target(df: pd.DataFrame, target_column: Union[str, List[str]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sépare les features (X) et la target (y) pour le ML.
    
    Args:
        df (pd.DataFrame): DataFrame complet
        target_column (str | list): Nom de la (ou des) colonne(s) cible(s)
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (X, y)
        
    Raises:
        KeyError: Si la colonne cible n'existe pas
    """
    
    if isinstance(target_column, str):
        target_column = [target_column]  # Convertit en liste pour uniformiser
    
    missing = [col for col in target_column if col not in df.columns]
    if missing:
        logging.error(f"Colonnes {missing} non trouvées dans le DataFrame")
        raise KeyError(f"Colonnes manquantes : {missing}")
        
    X = df.drop(columns=target_column)
    y = df[target_column]
    return X, y


# Gestion des doublons
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Supprime les doublons dans le dataset.
    
    Args:
        df (pd.DataFrame): Dataset d'entrée.
    
    Returns:
        pd.DataFrame: Dataset sans doublons.
    """
    initial_rows = df.shape[0]
    df = df.drop_duplicates()
    removed = initial_rows - df.shape[0]
    logging.info(f"{removed} doublons supprimés - Nouveau total : {df.shape[0]} lignes")
    return df


# Gestion des outliers
def detect_univariate_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Détecte et supprime les outliers univariés avec IQR."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    Q1 = df[numeric_cols].quantile(0.25)
    Q3 = df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df[numeric_cols] < (Q1 - 1.5 * IQR)) | (df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
    df_cleaned = df[~outliers]
    logging.info(f"{outliers.sum()} outliers univariés supprimés")
    return df_cleaned

def detect_multivariate_outliers(X: pd.DataFrame, contamination: float = 0.05) -> pd.DataFrame:
    """Détecte et supprime les outliers multivariés avec Isolation Forest."""
    iso = IsolationForest(contamination=contamination)
    outliers = iso.fit_predict(X)
    df_cleaned = X[outliers != -1]
    logging.info(f"{(outliers == -1).sum()} outliers multivariés supprimés")
    return df_cleaned


def winsorize_data(df: pd.DataFrame, limits: tuple = (0.01, 0.99)) -> pd.DataFrame:
    """Applique la winsorization aux colonnes numériques pour limiter les outliers."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = mstats.winsorize(df[col], limits=limits)
    logging.info(f"Winsorization appliquée avec limites {limits}")
    return df


# Gestion des données manquantes
def handle_missing_values(df: pd.DataFrame, threshold: float = 0.05) -> pd.DataFrame:
    """Gère les données manquantes avec une stratégie adaptative."""
    missing = df.isnull().sum()
    missing_prop = missing / df.shape[0]
    logging.info(f"Données manquantes :\n{missing[missing > 0]}\nProportions :\n{missing_prop[missing > 0]}")
    if missing.sum() == 0:
        return df
    if missing_prop.max() < 0.0005:
        df_cleaned = df.dropna()
        logging.info("Lignes supprimées (proportion < 0.05%)")
    elif missing_prop.max() < threshold:
        imputer = SimpleImputer(strategy='median')
        df_cleaned = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
        logging.info("Imputation avec médiane")
    else:
        imputer = KNNImputer(n_neighbors=5)
        df_cleaned = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
        logging.info("Imputation avec KNN")
    return df_cleaned


# Standardisation des données
def split_features_target(df: pd.DataFrame, target_column: str) -> tuple:
    """Sépare les features et la cible."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    logging.info(f"Features ({X.shape[1]}) et cible séparées")
    return X, y

def standardize_features(X: pd.DataFrame) -> tuple:
    """Standardise les features."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    logging.info("Features standardisées")
    return X_scaled, scaler


# Sauvegarde des données traitées
def save_preprocessed_data(X: np.ndarray, y: pd.DataFrame, scaler: StandardScaler, output_dir: str) -> None:
    """Sauvegarde les données prétraitées."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df_preprocessed = pd.concat([X_df, y.reset_index(drop=True)], axis=1)
    df_preprocessed.to_csv(output_dir / 'df_preprocessed.csv', index=False)

    joblib.dump(scaler, output_dir / 'scaler.joblib')
    logging.info(f"Données sauvegardées dans {output_dir}")


# Validation croisée pour tester les résultats
def validate_preprocessing(X, y):
    """Valide le prétraitement avec une validation croisée."""
    model = LogisticRegression(max_iter=1000)
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    logging.info(f"Accuracy moyenne : {scores.mean():.3f}")


# Pipeline de prétraitement
def preprocess_data(
    file_path: str,
    output_dir: str,
    target_column: Union[str, List[str]],
    remove_duplicates_flag: bool = False,
    outlier_method: str = 'none',  # 'univariate', 'multivariate', 'winsorize', 'none'
    winsorize_limits: tuple = (0.01, 0.99)
) -> Tuple[np.ndarray, pd.DataFrame, StandardScaler]:
    """
    Pipeline complet de prétraitement avec options configurables.
    
    Args:
        file_path (str): Chemin du fichier d'entrée
        output_dir (str): Répertoire de sortie
        target_column (str | list): Colonne(s) cible(s)
        remove_duplicates_flag (bool): Supprimer les doublons ou non
        outlier_method (str): Méthode pour les outliers
        winsorize_limits (tuple): Limites pour la winsorization
    
    Returns:
        Tuple[np.ndarray, pd.DataFrame, StandardScaler]: (X_scaled, y, scaler)
    """
    df = load_data(file_path)
    
    if remove_duplicates_flag:
        df = remove_duplicates(df)
    
    if outlier_method == 'univariate':
        df = detect_univariate_outliers(df)
    elif outlier_method == 'multivariate':
        X, y = split_features_target(df, target_column)
        X = detect_multivariate_outliers(X)
        df = pd.concat([X, y], axis=1)
    elif outlier_method == 'winsorize':
        df = winsorize_data(df, limits=winsorize_limits)
    elif outlier_method != 'none':
        logging.warning(f"Aucune méthode d'outliers.")
    
    df = handle_missing_values(df)
    
    X, y = split_features_target(df, target_column)
    X_scaled, scaler = standardize_features(X)
    save_preprocessed_data(X_scaled, y, scaler, output_dir)
    # validate_preprocessing(X_scaled, y.iloc[:, 0] if y.shape[1] > 1 else y)
    return X_scaled, y, scaler


if __name__ == "__main__":
    file_path = "./data/winequality-red.csv"
    output_dir = "./data/"
    # X_scaled, y, scaler = preprocess_data(file_path, output_dir, 'quality')
    X_scaled, y, scaler = preprocess_data(
        file_path=file_path,
        output_dir=output_dir,
        target_column='quality',
        remove_duplicates_flag=False,  # Conserver les doublons
        outlier_method='none',    # Utiliser la winsorization
        winsorize_limits=(0.01, 0.99) # Limites à 1% et 99%
    )
    print("Prétraitement terminé !")