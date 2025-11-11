"""
Utilitaires pour visualiser les résultats du modèle
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional


def plot_predictions_vs_targets(
    predictions: np.ndarray,
    targets: np.ndarray,
    variable_names: List[str],
    save_path: Optional[Path] = None,
    show_plot: bool = True
):
    """
    Affiche les prédictions vs vraies valeurs pour chaque variable.
    
    Args:
        predictions: Array (n_samples, n_variables)
        targets: Array (n_samples, n_variables)
        variable_names: Liste des noms de variables
        save_path: Chemin de sauvegarde optionnel
        show_plot: Afficher le plot
    """
    n_vars = predictions.shape[1]
    n_cols = 3
    n_rows = (n_vars + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_vars > 1 else [axes]
    
    for i in range(n_vars):
        ax = axes[i]
        
        pred_i = predictions[:, i]
        target_i = targets[:, i]
        
        # Filtrer les NaN
        valid_mask = ~(np.isnan(pred_i) | np.isnan(target_i))
        pred_valid = pred_i[valid_mask]
        target_valid = target_i[valid_mask]
        
        # Scatter plot
        ax.scatter(target_valid, pred_valid, alpha=0.5, s=10)
        
        # Ligne parfaite (y=x)
        min_val = min(target_valid.min(), pred_valid.min())
        max_val = max(target_valid.max(), pred_valid.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Parfait')
        
        # Statistiques
        mae_val = np.mean(np.abs(pred_valid - target_valid))
        r2_val = 1 - np.sum((pred_valid - target_valid)**2) / np.sum((target_valid - target_valid.mean())**2)
        
        ax.set_xlabel('Vraies valeurs')
        ax.set_ylabel('Prédictions')
        ax.set_title(f'{variable_names[i]}\nMAE: {mae_val:.3f}, R²: {r2_val:.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Cacher les axes inutilisés
    for i in range(n_vars, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Plot sauvegardé: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_residuals(
    residuals: np.ndarray,
    variable_names: List[str],
    save_path: Optional[Path] = None,
    show_plot: bool = True
):
    """
    Affiche les histogrammes des résidus pour chaque variable.
    
    Args:
        residuals: Array (n_samples, n_variables)
        variable_names: Liste des noms de variables
        save_path: Chemin de sauvegarde optionnel
        show_plot: Afficher le plot
    """
    n_vars = residuals.shape[1]
    n_cols = 3
    n_rows = (n_vars + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten() if n_vars > 1 else [axes]
    
    for i in range(n_vars):
        ax = axes[i]
        
        res_i = residuals[:, i]
        res_valid = res_i[~np.isnan(res_i)]
        
        ax.hist(res_valid, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(0, color='r', linestyle='--', label='Zéro')
        ax.axvline(res_valid.mean(), color='g', linestyle='--', label=f'Mean: {res_valid.mean():.3f}')
        
        ax.set_xlabel('Résidu (Prédit - Vrai)')
        ax.set_ylabel('Fréquence')
        ax.set_title(f'{variable_names[i]}\nStd: {res_valid.std():.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Cacher les axes inutilisés
    for i in range(n_vars, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Plot sauvegardé: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def print_prediction_stats(predictions_dict: Dict, variable_names: List[str]):
    """
    Affiche les statistiques détaillées des prédictions.
    
    Args:
        predictions_dict: Dict avec 'predictions', 'targets', 'residuals'
        variable_names: Liste des noms de variables
    """
    predictions = predictions_dict['predictions']
    targets = predictions_dict['targets']
    residuals = predictions_dict['residuals']
    
    print(f"\n{'='*70}")
    print(f"STATISTIQUES DES PRÉDICTIONS")
    print(f"{'='*70}")
    
    for i, var_name in enumerate(variable_names):
        pred_i = [predictions[j][i] for j in range(len(predictions))]
        target_i = [targets[j][i] for j in range(len(targets))]
        res_i = [residuals[j][i] for j in range(len(residuals))]
        
        # Filtrer NaN
        #valid_mask = ~(np.isnan(pred_i) | np.isnan(target_i))
        #pred_valid = pred_i[valid_mask]
        #target_valid = target_i[valid_mask]
        #res_valid = res_i[valid_mask]
        
        if len(pred_i) == 0:
            print(f"\n{var_name}: Aucune donnée valide")
            continue
        
        mae_val = np.mean(np.abs(res_i))
        rmse_val = np.sqrt(np.mean(res_i**2))
        r2_val = 1 - np.sum(res_i**2) / np.sum((target_i - target_i.mean())**2)
        
        print(f"\n{var_name}:")
        print(f"  MAE:  {mae_val:.4f}")
        print(f"  RMSE: {rmse_val:.4f}")
        print(f"  R²:   {r2_val:.4f}")
        print(f"  Résidu moyen: {res_i.mean():.4f} ± {res_i.std():.4f}")
        print(f"  Range prédictions: [{pred_i.min():.2f}, {pred_i.max():.2f}]")
        print(f"  Range vraies val.: [{target_i.min():.2f}, {target_i.max():.2f}]")

    print(f"{'='*70}\n")


if __name__ == "__main__":
    # Exemple d'utilisation
    #Load data from models/*.json

    #+current path
    file_path = Path().cwd() / "saved_models/MultiChannelCNN/MultiChannelCNN_ep10_loss239902.3875_20251111_180057.json"
    import json
    with open(file_path, 'r') as f:
        data = json.load(f)
        data = data['predictions']
    
    #Transform to numpy arrays
    predictions = np.array(data['predictions'])
    print(predictions.shape) # (n_samples, n_variables)
    print(predictions)
    targets = np.array(data['targets'])
    residuals = np.array(data['residuals'])

    variable_names = ['dd', 'ff', 'precip', 'hu', 'td', 't', 'psl']


    # Afficher les statistiques
    #print_prediction_stats(data, variable_names)
    
    # Tracer les prédictions vs vraies valeurs
    plot_predictions_vs_targets(
        predictions,
        targets,
        variable_names,
        show_plot=True
    )
    
    # Tracer les résidus
    plot_residuals(
        residuals,
        variable_names,
        show_plot=True
    )