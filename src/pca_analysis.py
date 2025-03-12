from sklearn.decomposition import PCA

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
import logging
from typing import List, Tuple, Union

from exploratory_analysis import split_features_target

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def perform_pca(X_scaled: np.ndarray, feature_names: List[str], n_components: int = 5) -> dict:
    """
    Effectue une analyse en composantes principales (PCA) sur les données prétraitées.

    Args:
        X_scaled (np.ndarray): Données prétraitées et standardisées (tableau numpy).
        feature_names (List[str]): Liste des noms des features.
        n_components (int, optional): Nombre de composantes principales à calculer. Par défaut, 5.

    Returns:
        dict: Dictionnaire contenant :
            - 'X_pca' : Données transformées en composantes principales.
            - 'explained_variance' : Variance expliquée par chaque composante.
            - 'total_variance' : Somme de la variance expliquée.
            - 'loadings' : DataFrame des contributions des features à chaque composante.
    """
    # Initialisation de la PCA avec le nombre de composantes spécifié
    pca = PCA(n_components=n_components)
    
    # Ajustement de la PCA aux données et transformation en composantes principales
    X_pca = pca.fit_transform(X_scaled)
    
    # Calcul de la variance expliquée par chaque composante
    explained_variance = pca.explained_variance_ratio_
    
    # Calcul de la variance totale expliquée
    total_variance = sum(explained_variance)
    
    # Création d’un DataFrame pour les "loadings" (contributions des features à chaque composante)
    loadings = pd.DataFrame(
        pca.components_.T,  # Transposé pour avoir les features en lignes et composantes en colonnes
        columns=[f'PC{i+1}' for i in range(n_components)],  # Noms des colonnes : PC1, PC2, etc.
        index=feature_names  # Noms des features comme index
    )
    
    # Visualisation de la variance expliquée par chaque composante
    plt.figure(figsize=(8, 6))
    plt.bar(range(1, n_components + 1), explained_variance)
    plt.title(f"Variance expliquée par les {n_components} premières composantes")
    plt.xlabel("Composante principale")
    plt.ylabel("Proportion de variance expliquée")
    plt.savefig(f"./figures/pca_variance_{n_components}.png")
    plt.clf()
    plt.close()  # Ferme la figure pour éviter des conflits dans des appels multiples
    
    # Retourne les résultats dans un dictionnaire pour une réutilisation facile
    return {
        'X_pca': X_pca,
        'explained_variance': explained_variance,
        'total_variance': total_variance,
        'loadings': loadings
    }


def plot_pca_correlation_circle(
    loadings: pd.DataFrame,
    components: Tuple[int, int] = (1, 2),
    figures_dir: str = "./figures"
) -> None:
    """
    Trace le cercle de corrélation pour visualiser les corrélations entre variables et composantes principales.

    Args:
        loadings (pd.DataFrame): DataFrame des contributions des features aux composantes (issu de perform_pca).
        components (Tuple[int, int], optional): Numéros des composantes à comparer (ex. (1, 2) pour PC1 vs PC2). Par défaut, (1, 2).
        figures_dir (str, optional): Répertoire où sauvegarder la figure. Par défaut, "./figures".
    """
    pc_x, pc_y = components  # Ex. : PC1 et PC2
    pc_x_label = f'PC{pc_x}'
    pc_y_label = f'PC{pc_y}'
    
    if pc_x > len(loadings.columns) or pc_y > len(loadings.columns):
        raise ValueError(f"Les composantes demandées ({pc_x}, {pc_y}) dépassent le nombre disponible ({len(loadings.columns)})")
    
    # Coordonnées des vecteurs pour chaque feature
    x = loadings[pc_x_label]
    y = loadings[pc_y_label]
    
    # Création du graphique
    plt.figure(figsize=(8, 8))
    plt.axhline(0, color='gray', lw=0.5)
    plt.axvline(0, color='gray', lw=0.5)
    
    # Cercle de référence (rayon 1)
    circle = plt.Circle((0, 0), 1, fill=False, color='black', linestyle='--')
    plt.gca().add_artist(circle)
    
    # Tracé des vecteurs
    for i, (xi, yi) in enumerate(zip(x, y)):
        plt.arrow(0, 0, xi, yi, color='b', alpha=0.5, head_width=0.05)
        plt.text(xi * 1.1, yi * 1.1, loadings.index[i], color='b', ha='center', va='center')
    
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.xlabel(f'{pc_x_label} ({loadings[pc_x_label].var():.2%} variance)')
    plt.ylabel(f'{pc_y_label} ({loadings[pc_y_label].var():.2%} variance)')
    plt.title(f'Cercle de corrélation : {pc_x_label} vs {pc_y_label}')
    plt.grid(True)
    
    # Sauvegarde
    figures_path = Path(figures_dir)
    figures_path.mkdir(parents=True, exist_ok=True)
    try:
        plt.savefig(figures_path / f"pca_correlation_circle_PC{pc_x}_PC{pc_y}.png")
        logging.info(f"Cercle de corrélation sauvegardé dans {figures_path / f'pca_correlation_circle_PC{pc_x}_PC{pc_y}.png'}")
    except Exception as e:
        logging.warning(f"Échec de la sauvegarde du cercle de corrélation : {e}")
    finally:
        plt.clf()
        plt.close()


def plot_pca_projection(
    X_pca: np.ndarray,
    y: Union[pd.Series, pd.DataFrame],
    components: Tuple[int, int] = (1, 2),
    figures_dir: str = "../figures",
    color_label: str = "Target"
) -> None:
    """
    Projette les données dans l’espace des composantes principales avec la cible en couleur.

    Args:
        X_pca (np.ndarray): Données transformées par PCA (issu de perform_pca).
        y (pd.Series | pd.DataFrame): Variable cible pour la couleur.
        components (Tuple[int, int], optional): Numéros des composantes à projeter. Par défaut, (1, 2).
        figures_dir (str, optional): Répertoire où sauvegarder la figure. Par défaut, "../figures".
        color_label (str, optional): Étiquette pour la barre de couleur. Par défaut, "Target".
    """
    pc_x, pc_y = components
    if pc_x > X_pca.shape[1] or pc_y > X_pca.shape[1]:
        raise ValueError(f"Les composantes demandées ({pc_x}, {pc_y}) dépassent le nombre disponible ({X_pca.shape[1]})")
    
    # Extraction des valeurs numériques de y, qu’il soit Series ou DataFrame
    y_values = y.iloc[:, 0].values if isinstance(y, pd.DataFrame) else y.values
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, pc_x - 1], X_pca[:, pc_y - 1], c=y_values, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label=color_label)
    plt.xlabel(f'PC{pc_x}')
    plt.ylabel(f'PC{pc_y}')
    plt.title(f'Projection des données sur PC{pc_x} vs PC{pc_y}')
    plt.grid(True)
    
    figures_path = Path(figures_dir)
    figures_path.mkdir(parents=True, exist_ok=True)
    try:
        plt.savefig(figures_path / f"pca_projection_PC{pc_x}_PC{pc_y}.png")
        logging.info(f"Projection sauvegardée dans {figures_path / f'pca_projection_PC{pc_x}_PC{pc_y}.png'}")
    except Exception as e:
        logging.warning(f"Échec de la sauvegarde de la projection : {e}")
    finally:
        plt.clf()
        plt.close()


if __name__ == "__main__":
    # Chargement des données prétraitées
    df = pd.read_csv("./data/df_preprocessed.csv")
    X_scaled, y = split_features_target(df, 'quality')
    
    # Extraction des noms des features (sans la cible 'quality')
    feature_names = df.drop('quality', axis=1).columns
    
    # Exécution de la PCA avec 5 composantes par défaut
    pca_results = perform_pca(X_scaled, feature_names, n_components=5)
    
    # Affichage des résultats pour inspection
    print("Variance expliquée par composante :", pca_results['explained_variance'])
    print("Total variance expliquée :", pca_results['total_variance'])
    print("\nContribution des features à PC1 :")
    print(pca_results['loadings']['PC1'].sort_values(ascending=False))
    print("Fin de l'Analyse en Composantes principales!")

    # Visualisation supplémentaire
    plot_pca_correlation_circle(pca_results['loadings'], components=(1, 2), figures_dir="./figures")
    plot_pca_projection(pca_results['X_pca'], y, components=(1, 2), figures_dir="./figures", color_label="quality")