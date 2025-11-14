"""
Utilitaires pour visualiser les résultats du modèle
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import argparse
import re
import math
from normalisation import denormalize_variable, denormalize_residual_variable

VARS_SHORT_TO_FULL = {
    "dd": "Direction du vent (°)",
    "ff": "Vitesse du vent (m/s)",
    "precip": "Précipitations (kg.m-²)",
    "hu": "Humidité (%)",
    "td": "Point de rosée (°C)",
    "t": "Température (°C)",
    "psl": "Pression au niveau de la mer (hPa)"
}


def parse_model_name(model_name: str) -> Dict[str, str]:
    """
    Parse le nom du modèle pour extraire les hyperparamètres.
    
    Format attendu: ModelName-wo-dropout-ds5-normalized-R2-precip-dd
    
    Returns:
        Dict avec les informations extraites
    """
    info = {
        'base_model': '',
        'dropout': 'Non spécifié',
        'downscaling': 'x1 (aucun)',
        'normalized': 'Non',
        'loss_function': 'R² (défaut)',
        'excluded_vars': 'Aucune'
    }
    
    # Séparer par tirets
    parts = model_name.split('-')
    
    if not parts:
        return info
    
    # Premier élément = nom du modèle de base
    info['base_model'] = parts[0]
    
    # Variables exclues (à la fin, après les marqueurs connus)
    excluded_vars = []
    known_markers = ['wo', 'dropout', 'ds', 'normalized', 'R2', 'MSE', 'MAE']
    
    # Parser les parties
    i = 1
    while i < len(parts):
        part = parts[i].lower()
        
        # Dropout
        if part == 'wo' and i + 1 < len(parts) and parts[i + 1].lower() == 'dropout':
            info['dropout'] = '0%'
            i += 2
        elif 'dropout' in part:
            # Chercher un nombre avant ou après
            match = re.search(r'(\d+)', part)
            if match:
                info['dropout'] = f"{match.group(1)}%"
            i += 1
        
        # Downscaling
        elif part.startswith('ds'):
            match = re.search(r'ds(\d+)', part)
            if match:
                info['downscaling'] = f"x{match.group(1)}"
            i += 1
        
        # Normalisation
        elif 'normalized' in part or 'norm' in part:
            info['normalized'] = 'Oui'
            i += 1
        
        # Fonction de coût
        elif part == 'r2':
            info['loss_function'] = 'Maximisation du R²'
            i += 1
        elif part == 'mse':
            info['loss_function'] = 'Minimisation du MSE'
            i += 1
        elif part == 'mae':
            info['loss_function'] = 'Minimisation du MAE'
            i += 1
        
        # Variables exclues (celles qui ne matchent aucun marqueur)
        elif part not in known_markers and part in ['dd', 'ff', 'precip', 'hu', 'td', 't', 'psl']:
            excluded_vars.append(part)
            i += 1
        else:
            i += 1
    
    # Formatter les variables exclues
    if excluded_vars:
        info['excluded_vars'] = ', '.join([VARS_SHORT_TO_FULL.get(v, v) for v in excluded_vars])
    
    return info


def create_hyperparameters_text(model_name: str, hyperparams: Optional[Dict] = None, dataset_size: Optional[int] = None) -> str:
    """
    Crée un texte formaté avec les hyperparamètres extraits du nom du modèle
    et les hyperparamètres enregistrés dans le JSON (si disponibles).
    """
    info = parse_model_name(model_name)

    # Valeurs par défaut
    epochs = None
    batch_size = None
    learning_rate = None
    optimizer = None

    if hyperparams:
        # Cherche diverses clés possibles
        epochs = hyperparams.get('num_epochs') or hyperparams.get('epochs')
        batch_size = hyperparams.get('batch_size')
        learning_rate = hyperparams.get('learning_rate') or hyperparams.get('lr')
        optimizer = hyperparams.get('optimizer') or hyperparams.get('optim')

    # Formattage propre
    def fmt_lr(x):
        if x is None:
            return "N/A"
        try:
            return f"{float(x):.2e}"
        except Exception:
            return str(x)

    text_lines = [
        f"Modèle: {info['base_model']}\n",
        f"Dropout: {info['dropout']}",
        f"Downscaling: {info['downscaling']}",
        f"Normalisation: {info['normalized']}",
        f"Fonction de coût: {info['loss_function']}",
        f"Variables exclues: {info['excluded_vars']}",
        "",
        f"Époques: {epochs if epochs is not None else 'N/A'}",
        f"Taille du batch: {batch_size if batch_size is not None else 'N/A'}",
        f"Learning rate: {fmt_lr(learning_rate)}",
        f"Taille du dataset: {dataset_size if dataset_size is not None else 'N/A'}",
        f"Optimiseur: {optimizer if optimizer is not None else 'N/A'}",
    ]
    return '\n'.join(text_lines)


def plot_predictions_vs_targets(
    model_name: str,
    predictions: np.ndarray,
    targets: np.ndarray,
    variable_names: List[str],
    save_path: Optional[Path] = None,
    show_plot: bool = True,
    hyperparams: Optional[Dict] = None,
    dataset_size: Optional[int] = None,
):
    """
    Affiche les prédictions vs vraies valeurs pour chaque variable.
    """
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
    if targets.ndim == 1:
        targets = targets.reshape(-1, 1)

    n_vars = predictions.shape[1]
    n_cols = 3
    # Réserver 1 case pour la légende
    n_rows = math.ceil((n_vars + 1) / n_cols)

    fig = plt.figure(figsize=(15, 5 * n_rows))
    fig.suptitle(f"Prédictions vs Vraies valeurs - {model_name}", fontsize=16, y=0.98)
    gs = fig.add_gridspec(n_rows, n_cols, hspace=0.3, wspace=0.3)

    # Case (0,0) = légende hyperparamètres
    ax_legend = fig.add_subplot(gs[0, 0])
    ax_legend.axis('off')
    hyperparams_text = create_hyperparameters_text(model_name, hyperparams=hyperparams, dataset_size=dataset_size)
    ax_legend.text(
        0.02, 0.98, hyperparams_text,
        ha='left', va='top', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
        transform=ax_legend.transAxes
    )

    # Créer les axes des graphes (toutes les cases sauf (0,0))
    graph_axes: List[plt.Axes] = []
    for r in range(n_rows):
        for c in range(n_cols):
            if r == 0 and c == 0:
                continue
            graph_axes.append(fig.add_subplot(gs[r, c]))

    # Tracer chaque variable
    for i in range(n_vars):
        ax = graph_axes[i]
        pred_i = predictions[:, i]
        target_i = targets[:, i]

        valid_mask = ~(np.isnan(pred_i) | np.isnan(target_i))
        pred_valid = pred_i[valid_mask]
        target_valid = target_i[valid_mask]

        # Dénormaliser
        pred_valid = denormalize_variable(variable_names[i], pred_valid)
        target_valid = denormalize_variable(variable_names[i], target_valid)

        ax.scatter(target_valid, pred_valid, alpha=0.5, s=10)

        if pred_valid.size > 0 and target_valid.size > 0:
            min_val = float(min(target_valid.min(), pred_valid.min()))
            max_val = float(max(target_valid.max(), pred_valid.max()))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Parfait')

            mae_val = float(np.mean(np.abs(pred_valid - target_valid)))
            denom = np.sum((target_valid - target_valid.mean())**2)
            r2_val = 1 - np.sum((pred_valid - target_valid)**2) / denom if denom > 0 else np.nan
        else:
            mae_val, r2_val = np.nan, np.nan

        ax.set_xlabel('Vraies valeurs')
        ax.set_ylabel('Prédictions')
        ax.set_title(f'{VARS_SHORT_TO_FULL.get(variable_names[i], variable_names[i])}\nMAE: {mae_val:.3f}, R²: {r2_val:.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Cacher axes inutilisés
    for j in range(n_vars, len(graph_axes)):
        graph_axes[j].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Plot sauvegardé: {save_path}")
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_residuals(
    model_name: str,
    residuals: np.ndarray,
    variable_names: List[str],
    save_path: Optional[Path] = None,
    show_plot: bool = True,
    hyperparams: Optional[Dict] = None,
    dataset_size: Optional[int] = None,
):
    """
    Affiche les histogrammes des résidus pour chaque variable.
    """
    if residuals.ndim == 1:
        residuals = residuals.reshape(-1, 1)

    n_vars = residuals.shape[1]
    n_cols = 3
    # Réserver 1 case pour la légende
    n_rows = math.ceil((n_vars + 1) / n_cols)

    fig = plt.figure(figsize=(15, 4 * n_rows))
    fig.suptitle(f"Histogrammes des résidus - {model_name}", fontsize=16, y=0.98)
    gs = fig.add_gridspec(n_rows, n_cols, hspace=0.3, wspace=0.3)

    # Case (0,0) = légende hyperparamètres
    ax_legend = fig.add_subplot(gs[0, 0])
    ax_legend.axis('off')
    hyperparams_text = create_hyperparameters_text(model_name, hyperparams=hyperparams, dataset_size=dataset_size)
    ax_legend.text(
        0.02, 0.98, hyperparams_text,
        ha='left', va='top', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
        transform=ax_legend.transAxes
    )

    # Axes pour histogrammes
    graph_axes: List[plt.Axes] = []
    for r in range(n_rows):
        for c in range(n_cols):
            if r == 0 and c == 0:
                continue
            graph_axes.append(fig.add_subplot(gs[r, c]))

    for i in range(n_vars):
        ax = graph_axes[i]
        res_i = residuals[:, i]
        res_valid = res_i[~np.isnan(res_i)]

        # Dénormaliser les résidus
        res_valid = denormalize_residual_variable(variable_names[i], res_valid)

        ax.hist(res_valid, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(0, color='r', linestyle='--', label='Zéro')
        if res_valid.size > 0:
            ax.axvline(res_valid.mean(), color='g', linestyle='--', label=f'Moyenne: {res_valid.mean():.3f}')

        ax.set_xlabel('Résidu (Prédit - Vrai)')
        ax.set_ylabel('Fréquence')
        ax.set_title(f'{VARS_SHORT_TO_FULL.get(variable_names[i], variable_names[i])}\nÉcart-type: {res_valid.std():.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    for j in range(n_vars, len(graph_axes)):
        graph_axes[j].axis('off')

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
        
        if len(pred_i) == 0:
            print(f"\n{var_name}: Aucune donnée valide")
            continue
        
        mae_val = np.mean(np.abs(res_i))
        rmse_val = np.sqrt(np.mean(np.array(res_i)**2))
        r2_val = 1 - np.sum(np.array(res_i)**2) / np.sum((np.array(target_i) - np.mean(target_i))**2)
        
        print(f"\n{var_name}:")
        print(f"  MAE:  {mae_val:.4f}")
        print(f"  RMSE: {rmse_val:.4f}")
        print(f"  R²:   {r2_val:.4f}")
        print(f"  Résidu moyen: {np.mean(res_i):.4f} ± {np.std(res_i):.4f}")
        print(f"  Range prédictions: [{np.min(pred_i):.2f}, {np.max(pred_i):.2f}]")
        print(f"  Range vraies val.: [{np.min(target_i):.2f}, {np.max(target_i):.2f}]")

    print(f"{'='*70}\n")


def find_latest_json(root_dir: Path) -> Optional[Path]:
    """
    Trouve le fichier JSON le plus récent (modification) dans root_dir et ses sous-dossiers.
    
    Args:
        root_dir: Dossier racine à parcourir
        
    Returns:
        Path du fichier JSON le plus récent, ou None si aucun trouvé
    """
    json_files = list(root_dir.rglob("*.json"))
    
    if not json_files:
        return None
    
    # Trier par date de modification (plus récent en premier)
    latest = max(json_files, key=lambda p: p.stat().st_mtime)
    return latest


def find_all_jsons(root_dir: Path) -> List[Path]:
    """
    Trouve tous les fichiers JSON dans root_dir et ses sous-dossiers.
    
    Args:
        root_dir: Dossier racine à parcourir
        
    Returns:
        Liste des chemins de fichiers JSON trouvés
    """
    return list(root_dir.rglob("*.json"))


def process_single_json(
    json_path: Path,
    show_plot: bool = False
) -> bool:
    """
    Traite un seul fichier JSON et génère les plots dans le même dossier.
    
    Args:
        json_path: Chemin du fichier JSON
        variable_names: Liste des noms de variables
        show_plot: Afficher les plots (False pour batch)
        
    Returns:
        True si succès, False sinon
    """
    try:
        # Charger les données
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Vérifier la structure
        if 'predictions' in data:
            pred_data = data['predictions']
            predictions = np.array(pred_data['predictions'])
            targets = np.array(pred_data['targets'])
            residuals = np.array(pred_data['residuals'])
        else:
            return False

        # Hyperparamètres (optionnels) et taille dataset
        hyperparams = data.get('hyperparameters', None)
        ds_size = None
        try:
            ds_size = int(data.get('dataset_size', 0)) or int(predictions.shape[0])
        except Exception:
            ds_size = int(predictions.shape[0])

        var_names = data.get('output_parameters', None)
        if var_names is None:
            output_dim = predictions.shape[1]
            if output_dim == 7:
                var_names = ['dd', 'ff', 'precip', 'hu', 'td', 't', 'psl']
            elif output_dim == 6:
                var_names = ['dd', 'ff', 'hu', 'td', 't', 'psl']
            elif output_dim == 5:
                var_names = ['ff', 'hu', 'td', 't', 'psl']
        
        
        
        
        # Chemins de sauvegarde (même dossier que le JSON)
        save_dir = json_path.parent
        base_name = json_path.stem
        save_pred_path = save_dir / f"{base_name}_predictions.png"
        save_resid_path = save_dir / f"{base_name}_residuals.png"
        
        # Générer les plots
        plot_predictions_vs_targets(
            data.get('model_name', 'Unknown Model'),
            predictions,
            targets,
            var_names,
            save_path=save_pred_path,
            show_plot=show_plot,
            hyperparams=hyperparams,
            dataset_size=ds_size,
        )

        plot_residuals(
            data.get('model_name', 'Unknown Model'),
            residuals,
            var_names,
            save_path=save_resid_path,
            show_plot=show_plot,
            hyperparams=hyperparams,
            dataset_size=ds_size,
        )
        
        print(f"  ✅ {json_path.name}")
        return True
        
    except Exception as e:
        print(f"  ❌ Erreur avec {json_path.name}: {e}")
        return False


def calculate_r2_score(json_path: Path) -> Optional[float]:
    """
    Calcule le R² moyen à partir des prédictions dans le JSON.
    
    Args:
        json_path: Chemin du fichier JSON
        
    Returns:
        R² moyen ou None si impossible à calculer
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        if 'predictions' not in data:
            return None
        
        pred_data = data['predictions']
        predictions = np.array(pred_data['predictions'])
        targets = np.array(pred_data['targets'])
        
        # Calculer R² pour chaque variable
        r2_scores = []
        for i in range(predictions.shape[1]):
            pred_i = predictions[:, i]
            target_i = targets[:, i]
            
            valid_mask = ~(np.isnan(pred_i) | np.isnan(target_i))
            pred_valid = pred_i[valid_mask]
            target_valid = target_i[valid_mask]
            
            if len(pred_valid) > 0:
                denom = np.sum((target_valid - target_valid.mean())**2)
                if denom > 0:
                    r2 = 1 - np.sum((pred_valid - target_valid)**2) / denom
                    r2_scores.append(r2)
        
        if r2_scores:
            return float(np.mean(r2_scores))
        return None
    except Exception:
        return None


def find_best_model(root_dir: Path, metric: str = 'r2', batch_only: bool = False) -> Optional[Tuple[Path, float, Optional[float]]]:
    """
    Trouve le meilleur modèle dans root_dir et sous-dossiers selon la métrique choisie.
    
    Args:
        root_dir: Dossier racine à parcourir
        metric: 'mse' pour test_loss minimal, 'r2' pour R² maximal
        batch_only: Si True, ne considère que les modèles dont le nom commence par "BATCH-"
        
    Returns:
        Tuple (chemin du JSON, test_loss, R²) ou None si aucun trouvé
    """
    json_files = list(root_dir.rglob("*.json"))
    
    if not json_files:
        return None
    
    # Filtrer pour ne garder que les BATCH si demandé
    if batch_only:
        json_files = [
            f for f in json_files 
            if f.stem.startswith("BATCH-") or "BATCH-" in f.name
        ]
        
        if not json_files:
            return None
    
    best_path = None
    best_mse = float('inf')
    best_r2 = -float('inf')
    
    for json_path in json_files:
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Vérifier aussi le model_name dans le JSON
            if batch_only:
                model_name = data.get('model_name', '')
                if not model_name.startswith('BATCH-'):
                    continue
            
            test_loss = data.get('test_loss')
            
            if metric == 'mse':
                # Chercher le MSE le plus faible
                if test_loss is not None and isinstance(test_loss, (int, float)):
                    if test_loss < best_mse:
                        best_mse = test_loss
                        best_path = json_path
            
            elif metric == 'r2':
                # Calculer le R² et chercher le plus élevé
                r2_score = calculate_r2_score(json_path)
                if r2_score is not None:
                    if r2_score > best_r2:
                        best_r2 = r2_score
                        best_mse = test_loss if test_loss is not None else float('nan')
                        best_path = json_path
        
        except Exception:
            continue
    
    if best_path is None:
        return None
    
    if metric == 'mse':
        return (best_path, best_mse, None)
    else:
        return (best_path, best_mse, best_r2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualisation des prédictions du modèle")
    parser.add_argument('--file', type=str, default=None,
                        help="Chemin vers le fichier JSON contenant les prédictions")
    parser.add_argument('--dir', type=str, default="saved_models",
                        help="Dossier où chercher le(s) JSON (défaut: saved_models)")
    parser.add_argument('--all', action='store_true',
                        help="Traiter TOUS les JSON du dossier (génère PNG à côté de chaque JSON)")
    parser.add_argument('--best', action='store_true',
                        help="Afficher automatiquement le meilleur modèle")
    parser.add_argument('--metric', type=str, choices=['mse', 'r2'], default='mse',
                        help="Métrique pour --best: 'mse' (test_loss minimal) ou 'r2' (R² maximal)")
    parser.add_argument('--batch-only', action='store_true',
                        help="Avec --best, ne considérer que les modèles BATCH-*")
    parser.add_argument('--variables', type=str, nargs='+', 
                        default=['dd', 'ff', 'precip', 'hu', 'td', 't', 'psl'],
                        help="Liste des noms de variables")
    parser.add_argument('--save-dir', type=str, default=None,
                        help="Dossier où sauvegarder les plots (optionnel, ignoré si --all)")
    parser.add_argument('--no-show', action='store_true',
                        help="Ne pas afficher les plots (seulement sauvegarder)")
    
    args = parser.parse_args()
    
    # Mode batch: traiter tous les JSON
    if args.all:
        root_dir = Path(args.dir)
        if not root_dir.exists():
            print(f"❌ Dossier non trouvé: {root_dir}")
            exit(1)
        
        json_files = find_all_jsons(root_dir)
        if not json_files:
            print(f"❌ Aucun fichier JSON trouvé dans {root_dir}")
            exit(1)
        
        print(f"📂 {len(json_files)} fichiers JSON trouvés")
        print(f"🔄 Traitement en cours...\n")
        
        success_count = 0
        for json_path in json_files:
            if process_single_json(json_path, show_plot=False):
                success_count += 1
        
        print(f"\n✅ Terminé: {success_count}/{len(json_files)} fichiers traités avec succès")
        exit(0)
    
    # Mode best: afficher le meilleur modèle
    if args.best:
        root_dir = Path(args.dir)
        if not root_dir.exists():
            print(f"❌ Dossier non trouvé: {root_dir}")
            exit(1)
        
        result = find_best_model(root_dir, metric=args.metric, batch_only=args.batch_only)
        if result is None:
            filter_msg = " (filtré sur BATCH-*)" if args.batch_only else ""
            print(f"❌ Aucun modèle trouvé dans {root_dir}{filter_msg}")
            exit(1)
        
        file_path, mse_val, r2_val = result
        
        batch_tag = " [BATCH]" if args.batch_only else ""
        if args.metric == 'mse':
            print(f"🏆 Meilleur modèle (MSE minimal){batch_tag}: {file_path}")
            print(f"   Test Loss (MSE): {mse_val:.6f}")
        else:
            print(f"🏆 Meilleur modèle (R² maximal){batch_tag}: {file_path}")
            print(f"   R² moyen: {r2_val:.6f}")
            if not np.isnan(mse_val):
                print(f"   Test Loss (MSE): {mse_val:.6f}")
        print()
    
    # Mode single: traiter un seul fichier (comportement original)
    elif args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"❌ Fichier non trouvé: {file_path}")
            exit(1)
    else:
        # Chercher le dernier JSON dans le dossier
        root_dir = Path(args.dir)
        if not root_dir.exists():
            print(f"❌ Dossier non trouvé: {root_dir}")
            exit(1)
        
        file_path = find_latest_json(root_dir)
        if file_path is None:
            print(f"❌ Aucun fichier JSON trouvé dans {root_dir}")
            exit(1)
        
        print(f"📂 Fichier le plus récent trouvé: {file_path}")
    
    # Charger les données
    print(f"📖 Chargement: {file_path}")
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Vérifier la structure
        if 'predictions' in data:
            pred_data = data['predictions']
            predictions = np.array(pred_data['predictions'])
            targets = np.array(pred_data['targets'])
            residuals = np.array(pred_data['residuals'])
        else:
            print(f"❌ Structure JSON non reconnue")
            exit(1)

        hyperparams = data.get('hyperparameters', None)
        try:
            dataset_size = int(data.get('dataset_size', 0)) or int(predictions.shape[0])
        except Exception:
            dataset_size = int(predictions.shape[0])
    except Exception as e:
        print(f"❌ Erreur de chargement: {e}")
        exit(1)
    
    # Préparer les chemins de sauvegarde
    save_pred_path = None
    save_resid_path = None
    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_pred_path = save_dir / f"predictions_{file_path.stem}.png"
        save_resid_path = save_dir / f"residuals_{file_path.stem}.png"
    
    # Variables (détecter automatiquement si possible)
    variable_names = data.get('output_parameters', args.variables)
    if predictions.shape[1] != len(variable_names):
        print(f"⚠️  Nombre de variables ne correspond pas ({predictions.shape[1]} vs {len(variable_names)})")
        print(f"   Utilisation de noms génériques...")
        variable_names = [f"var_{i}" for i in range(predictions.shape[1])]
    
    # Tracer les prédictions vs vraies valeurs
    if targets is not None:
        print(f"\n📊 Génération du plot prédictions vs targets...")
        plot_predictions_vs_targets(
            data.get('model_name', 'Unknown Model'),
            predictions,
            targets,
            variable_names,
            save_path=save_pred_path,
            show_plot=not args.no_show,
            hyperparams=hyperparams,
            dataset_size=dataset_size,
        )

    # Tracer les résidus
    if residuals is not None:
        print(f"📊 Génération du plot des résidus...")
        plot_residuals(
            data.get('model_name', 'Unknown Model'),
            residuals,
            variable_names,
            save_path=save_resid_path,
            show_plot=not args.no_show,
            hyperparams=hyperparams,
            dataset_size=dataset_size,
        )
    
    print(f"\n✅ Terminé!")