from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from typing import Tuple, Union, List
import logging


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


def dataset_summary(X: pd.DataFrame, y: pd.DataFrame, figures_path: Union[str, Path] = "./figures", classification_threshold: int = 10) -> None:
    """
    Affiche un rapport complet du dataset et génère des visualisations.
    
    Args:
        X (pd.DataFrame): Features
        y (pd.DataFrame): Target(s)
        figures_path (str/Path): Dossier de stockage des figures
    """

    # Configuration initiale
    figures_path = Path(figures_path)
    figures_path.mkdir(parents=True, exist_ok=True)
    df = pd.concat([X, y], axis=1)
    
    # 1. Affichage des métadonnées de base
    logging.info("\n=== Aperçu initial ===")
    print(f"\n{'-'*40}\nPremières lignes :\n{'-'*40}")
    print(df.head())
    
    print(f"\n{'-'*40}\nInformations structurelles :\n{'-'*40}")
    df.info()
    
    print(f"\n{'-'*40}\nStatistiques descriptives :\n{'-'*40}")
    print(df.describe(include='all'))
    
    # 2. Analyse des valeurs manquantes
    missing_values = df.isnull().sum()
    print(f"\n{'-'*40}\nValeurs manquantes :\n{'-'*40}")
    if not missing_values[missing_values > 0].empty:
        print(missing_values[missing_values > 0])
    else:
        print("Aucune valeur manquante !")
    
    # 3. Dimensions des données
    print(f"\n{'-'*40}\nDimensions :\n{'-'*40}")
    print(f"Features (X): {X.shape}\nTarget (y): {y.shape}")
    
    # 4. Analyse de chaque colonne target
    logging.info("\n=== Analyse des targets ===")
    for target_col in y.columns:
        plt.figure(figsize=(10, 6))
        
        # Détection automatique du type de problème
        unique_vals = y[target_col].nunique()
        is_classification = (unique_vals <= classification_threshold) or (y[target_col].dtype == 'object')
        
        if is_classification:
            # Visualisation des classes avec pourcentages
            plot = sns.countplot(x=y[target_col], hue=y[target_col], palette="viridis", legend=False)
            plt.title(f"Répartition des classes - {target_col}")
            
            total = len(y)
            for p in plot.patches:
                percentage = f'{100 * p.get_height()/total:.1f}%'
                plot.annotate(percentage, 
                            (p.get_x() + p.get_width()/2., p.get_height()), 
                            ha='center', va='center', 
                            xytext=(0, 5), 
                            textcoords='offset points')
        else:
            # Visualisation de la distribution continue
            sns.histplot(y[target_col], kde=True, bins=20)
            plt.title(f"Distribution - {target_col}")
        
        plt.xlabel(target_col)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(figures_path / f"target_{target_col}_distribution.png", dpi=120)
        plt.close()


def explore_dataset(X: pd.DataFrame, y: pd.DataFrame, figures_path: Union[str, Path] = "./figures", classification_threshold: int = 10) -> None:
    """
    Analyse exploratoire complète du dataset.
    
    Args:
        X (pd.DataFrame): Features (colonnes numériques attendues)
        y (pd.DataFrame): Target (une ou plusieurs colonnes, numériques ou catégorielles)
        figures_path (str/Path): Dossier pour sauvegarder les figures
        classification_threshold (int): Seuil pour déterminer si une target est catégorielle (fixé à 10 par défaut)
    """
    figures_path = Path(figures_path)

    # Création du dossier de sortie
    figures_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Matrice de corrélation
    logging.info("\n=== Corrélations ===")
    df = pd.concat([X, y], axis=1)
    corr_matrix = df.corr()
    
    plt.figure(figsize=(12, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', fmt=".2f", 
                annot_kws={"size": 8}, cbar_kws={"shrink": 0.8})
    plt.title("Matrice de corrélation", pad=20)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(figures_path/f"correlation_matrix.png")
    plt.clf()
    plt.close()

    # 2. Distributions des features avec tests de normalité
    logging.info("\n=== Distributions des features ===")
    for col in X.columns:
        # Double visualisation
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        
        # Histogramme + KDE
        sns.histplot(X[col], kde=True, ax=axs[0], color='skyblue')
        axs[0].set_title(f"Distribution de {col}")
        
        # QQ-Plot
        stats.probplot(X[col], plot=axs[1])
        axs[1].set_title(f"QQ-Plot de {col}")
        
        plt.tight_layout()
        plt.savefig(figures_path/f"{col}_distribution.png", bbox_inches='tight')
        plt.clf()
        plt.close()

    # 3. Relations features-targets
    logging.info("\n=== Relations features-targets ===")
    for target_col in y.columns:
        target_type = 'classification' if (y[target_col].nunique() <= classification_threshold) else 'regression'
        
        for feature_col in X.columns:
            plt.figure(figsize=(10, 6))
            
            if target_type == 'classification':
                # Violin plot pour classification
                sns.violinplot(x=y[target_col], y=X[feature_col], hue=y[target_col],
                              palette="viridis", legend=False, cut=0)
                plt.title(f"Distribution de {feature_col} par classe de {target_col}")
            else:
                # Regression plot avec intervalle de confiance
                sns.regplot(x=X[feature_col], y=y[target_col], 
                           scatter_kws={'alpha':0.3, 'color':'slategray'},
                           line_kws={'color':'crimson'}, ci=95)
                plt.title(f"Relation {feature_col} vs {target_col}")
            
            plt.tight_layout()
            plt.savefig(figures_path/f"relation_{feature_col}_vs_{target_col}.png", dpi=120)
            plt.clf()
            plt.close()

    # 4. Analyse multivariée sélective
    if len(y.columns) == 1:  # Pour éviter les visualisations trop complexes
        target_col = y.columns[0]
        top_features = corr_matrix[target_col].abs().sort_values(ascending=False).index[1:4]
        
        # Pairplot ciblé
        sns.pairplot(df[top_features.tolist() + [target_col]], 
                    diag_kind='kde',
                    plot_kws={'alpha':0.5, 'edgecolor':'none'},
                    diag_kws={'fill':True})
        plt.suptitle("Relations clés avec la target", y=1.02)
        plt.savefig(figures_path/f"key_relationships.png", bbox_inches='tight')
        plt.clf()
        plt.close()
    
    # 5. Boxplots simples des features (superflus)
    logging.info("\n=== Analyse des features ===")
    for col in X.columns:
        plt.figure(figsize=(10, 4))
        sns.boxplot(x=col, data=X)
        plt.title(f"Distribution de {col}")
        plt.savefig(figures_path/f"{col}_boxplot.png")
        plt.clf()
        plt.close()

    # 6. Diagrammes de dispersion ciblés (version multi-target)
    logging.info("\n=== Diagrammes de dispersion ciblés ===")

    for target_col in y.columns:
        try:
            # Sélection des 2 features les plus corrélées avec cette target
            top_features = corr_matrix[target_col].abs().sort_values(ascending=False).index[1:3]
            
            if len(top_features) >= 2:
                plt.figure(figsize=(10, 6))
                
                # Détermination du type de palette
                unique_classes = y[target_col].nunique()
                palette = "viridis" if (unique_classes <= classification_threshold) or (y[target_col].dtype == 'object') else "plasma"
                
                # Création du scatter plot
                scatter = sns.scatterplot(x=top_features[0], y=top_features[1], hue=target_col,
                                        data=df, palette=palette, alpha=0.7)
                
                # Optimisation de la légende
                handles, labels = scatter.get_legend_handles_labels()
                plt.legend(handles=handles[1:], title=target_col, bbox_to_anchor=(1.05, 1), loc='upper left')
                
                # Personnalisation du titre
                plt.title(f"Interaction {top_features[0]} et {top_features[1]}\nColorée par {target_col}", pad=20)
                plt.xlabel(top_features[0], fontweight='bold')
                plt.ylabel(top_features[1], fontweight='bold')
                
                plt.savefig(figures_path/f"scatter_{target_col}.png", bbox_inches='tight', dpi=120)
                plt.clf()
                plt.close()
                
        except Exception as e:
            logging.error(f"Erreur avec la target {target_col} : {str(e)}")
            continue

    # 7. Détection des outliers avec boxenplot
    plt.figure(figsize=(12, 6))
    sns.boxenplot(data=X, palette="Set3", orient="h")
    plt.title("Distribution des features avec détection d'outliers (boxenplot)")
    plt.tight_layout()
    plt.savefig(figures_path/f"outliers_detection.png")
    plt.clf()
    plt.close()


# Exemple d'utilisation :
if __name__ == "__main__":
    
    try:
        df = load_data("./data/winequality-red.csv")
        X, y = split_features_target(df, "quality")
        dataset_summary(X, y)
        explore_dataset(X, y)
        logging.info("Analyse exploratoire terminée !")
    except Exception as e:
        logging.error(f"Échec de l'analyse : {e}")