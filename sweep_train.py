import argparse
import itertools
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import gc

import h5py
import numpy as np
import torch
from torch.utils.data import TensorDataset

from models.model_multiimg_multidates import MultiChannelCNN
from training.train import train_model
from training.save import save_model_checkpoint
from training.normalisation import (
    normalize_dd, normalize_ff, normalize_precip, normalize_hu,
    normalize_td, normalize_t, normalize_psl
)

# Variables à extraire (sans dd et precip)
ALL_VARS = ['ff', 'hu', 'td', 't', 'psl']
EXCLUDED_VARS = ['dd', 'precip']  # Variables exclues
# Indices des variables dans le dataset original [dd, ff, precip, hu, td, t, psl]
VAR_INDICES = [1, 3, 4, 5, 6]  # ff, hu, td, t, psl
# Fonctions de normalisation correspondantes
NORM_FUNCS = [
    normalize_ff,    # ff (indice 1)
    normalize_hu,    # hu (indice 3)
    normalize_td,    # td (indice 4)
    normalize_t,     # t (indice 5)
    normalize_psl    # psl (indice 6)
]

# Cache pour éviter de recharger les mêmes datasets
_DATASET_CACHE: Dict[Tuple[int, str, int], Tuple[torch.Tensor, torch.Tensor]] = {}


def load_dataset_h5(dataset_dir: Path, year: int, station: str, ds: int, use_cache: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Charge X, y depuis: meteonet_SE_<year>_<station>_ds<ds>.h5
    X attendu (B, n_deltas, n_types, H, W), y (B, 7)
    Avec cache optionnel pour éviter rechargements.
    """
    cache_key = (year, station, ds)
    if use_cache and cache_key in _DATASET_CACHE:
        print(f"[CACHE] Dataset {year} ds{ds} déjà en mémoire")
        return _DATASET_CACHE[cache_key]
    
    fpath = dataset_dir / f"meteonet_SE_{year}_{station}_ds{ds}.h5"
    if not fpath.exists():
        raise FileNotFoundError(f"Fichier introuvable: {fpath}")
    print(f"Chargement dataset: {fpath}")
    with h5py.File(fpath, 'r') as f:
        X = torch.from_numpy(f['images'][:].astype(np.float32))
        y = torch.from_numpy(f['labels'][:].astype(np.float32))
    
    if use_cache:
        _DATASET_CACHE[cache_key] = (X, y)
    
    return X, y


def prep_inputs_multichannelcnn(X: torch.Tensor) -> torch.Tensor:
    """
    MultiChannelCNN attend (B, C, H, W) avec C = n_types * n_deltas.
    Dataset brut: (B, n_deltas, n_types, H, W)
    """
    if X.dim() != 5:
        raise ValueError(f"X 5D attendu, reçu {tuple(X.shape)}")
    # (B, n_deltas, n_types, H, W) -> (B, n_types, n_deltas, H, W)
    X = X.permute(0, 2, 1, 3, 4)
    # Aplatir canaux: (B, n_types * n_deltas, H, W)
    X = X.reshape(X.size(0), X.size(1) * X.size(2), X.size(3), X.size(4))
    return X


def prep_targets(y: torch.Tensor) -> torch.Tensor:
    """
    Nettoie NaN, normalise et extrait uniquement les colonnes ALL_VARS.
    y original: (B, 7) avec ordre [dd, ff, precip, hu, td, t, psl]
    y retourné: (B, 5) avec ordre [ff, hu, td, t, psl]
    """
    if y.dim() != 2:
        raise ValueError(f"y 2D attendu, reçu {tuple(y.shape)}")
    y = y.clone()

    # Remplacer NaN par 0
    y = torch.nan_to_num(y, nan=0.0)

    # Normaliser toutes les colonnes d'abord (dans l'ordre original)
    y[:, 0] = normalize_dd(y[:, 0])      # dd
    y[:, 1] = normalize_ff(y[:, 1])      # ff
    y[:, 2] = normalize_precip(y[:, 2])  # precip
    y[:, 3] = normalize_hu(y[:, 3])      # hu
    y[:, 4] = normalize_td(y[:, 4])      # td
    y[:, 5] = normalize_t(y[:, 5])       # t
    y[:, 6] = normalize_psl(y[:, 6])     # psl

    # Extraire uniquement les colonnes voulues: ff, hu, td, t, psl (indices 1, 3, 4, 5, 6)
    y = torch.cat([y[:, 1:2], y[:, 3:]], dim=1)  # [ff] + [hu, td, t, psl]
    
    return y


def make_model(n_types: int, n_deltas: int, output_dim: int, dropout_p: float) -> MultiChannelCNN:
    """
    Instancie MultiChannelCNN avec un dropout paramétrable.
    """
    model = MultiChannelCNN(n_types=n_types, n_deltas=n_deltas, output_dim=output_dim, dropout_p=float(dropout_p))
    return model


def model_already_trained(run_name: str) -> bool:
    """
    Vérifie si un modèle avec ce nom existe déjà dans saved_models/ ou runs/
    Cherche les .pth et .json correspondants
    """
    search_dirs = [Path("saved_models"), Path("runs")]
    
    for base_dir in search_dirs:
        if not base_dir.exists():
            continue
        
        # Chercher récursivement les fichiers avec ce nom (stem)
        pth_files = list(base_dir.rglob(f"{run_name}*.pth"))
        json_files = list(base_dir.rglob(f"{run_name}*.json"))
        
        if pth_files or json_files:
            return True
    
    return False


def run_one_training(
    dataset_dir: Path,
    station: str,
    base_year: int,
    finetune_years: List[int],
    ds: int,
    epochs: int,
    batch_size: int,
    lr: float,
    dropout_p: float,
    save_predictions: bool = False,
    skip_existing: bool = True
):
    # Nom modèle avec variables exclues
    dropout_tag = "wo-dropout" if dropout_p == 0 else f"dropout{int(round(dropout_p*100))}"
    excluded_tag = "-".join(EXCLUDED_VARS)  # "dd-precip"
    run_name = "MCNN"  # placeholder; la fonction save_model_checkpoint reconstruira un nom compact

    # Vérifier si déjà entraîné
    if skip_existing and model_already_trained(run_name):
        print(f"[SKIP] {run_name} déjà entraîné")
        
        # Vérifier aussi les finetunes
        all_skipped = True
        for yfin in finetune_years:
            ft_name = f"{run_name}_finetune{yfin}"
            if not model_already_trained(ft_name):
                all_skipped = False
                break
        
        if all_skipped:
            print(f"[SKIP] Tous les finetunes également présents")
            return
        else:
            print(f"[INFO] Certains finetunes manquants, chargement du modèle de base...")
            # TODO: Charger le modèle existant pour faire les finetunes manquants
            # Pour l'instant, on skip complètement
            return

    # Charger base (avec cache)
    X_raw, y_raw = load_dataset_h5(dataset_dir, base_year, station, ds, use_cache=True)
    
    # Préparer X pour MultiChannelCNN
    if X_raw.dim() != 5:
        raise ValueError(f"Format X inattendu: {tuple(X_raw.shape)} (attendu 5D)")
    _, n_deltas, n_types, _, _ = X_raw.shape
    X = prep_inputs_multichannelcnn(X_raw.clone())  # clone pour ne pas modifier le cache
    y = prep_targets(y_raw.clone())  # Extrait et normalise ff, hu, td, t, psl
    output_dim = y.shape[1]  # Devrait être 5

    del X_raw, y_raw  # libérer mémoire

    print(f"[{run_name}] X shape: {X.shape}, y shape: {y.shape}, output_dim: {output_dim}")

    # Créer modèle
    model = make_model(n_types=n_types, n_deltas=n_deltas, output_dim=output_dim, dropout_p=dropout_p)
    print(f"[{run_name}] Modèle créé, début de l'entraînement...")
    
    # Dataset et train
    dataset = TensorDataset(X, y)
    trained_model, optimizer, metrics = train_model(
        name=run_name,
        model=model,
        dataset=dataset,
        num_epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        return_predictions=save_predictions
    )

    # Sauvegarde checkpoint base
    try:
        save_model_checkpoint(
            model=trained_model,
            optimizer=optimizer,
            model_name=run_name,
            epoch=epochs,
            train_loss=metrics.get('train_loss') if isinstance(metrics, dict) else None,
            val_loss=metrics.get('val_loss') if isinstance(metrics, dict) else None,
            test_loss=metrics.get('test_loss') if isinstance(metrics, dict) else None,
            hyperparameters={
                "model": "MultiChannelCNN",
                "dropout": dropout_p,
                "downscaling": ds,
                "normalized": True,
                "learning_rate": lr,
                "batch_size": batch_size,
                "num_epochs": epochs,
                "base_year": base_year,
                "station": station,
                "output_dim": output_dim,
                "excluded_variables": EXCLUDED_VARS,
            },
            predictions=(metrics.get('predictions') if (isinstance(metrics, dict) and save_predictions) else None),
            output_parameters=ALL_VARS,  # Ajouter les noms des variables
            include_timestamp=True
        )
    except Exception as e:
        print(f"! Avertissement: échec sauvegarde checkpoint base: {e}")

    # Libération mémoire
    del X, y, dataset
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()

    # Finetune successifs
    current_model = trained_model
    for yfin in finetune_years:
        ft_name = f"{run_name}_finetune{yfin}"
        
        # Vérifier si ce finetune existe déjà
        if skip_existing and model_already_trained(ft_name):
            print(f"[SKIP] {ft_name} déjà entraîné")
            continue
        
        try:
            Xf_raw, yf_raw = load_dataset_h5(dataset_dir, yfin, station, ds, use_cache=True)
            Xf = prep_inputs_multichannelcnn(Xf_raw.clone())
            yf = prep_targets(yf_raw.clone())
            ds_finetune = TensorDataset(Xf, yf)

            current_model, optimizer, ft_metrics = train_model(
                name=ft_name,
                model=current_model,
                dataset=ds_finetune,
                num_epochs=epochs,          # on garde mêmes epochs pour finetune
                batch_size=batch_size,
                learning_rate=lr,
                return_predictions=save_predictions
            )

            save_model_checkpoint(
                model=current_model,
                optimizer=optimizer,
                model_name=ft_name,
                epoch=epochs,
                train_loss=ft_metrics.get('train_loss') if isinstance(ft_metrics, dict) else None,
                val_loss=ft_metrics.get('val_loss') if isinstance(ft_metrics, dict) else None,
                test_loss=ft_metrics.get('test_loss') if isinstance(ft_metrics, dict) else None,
                hyperparameters={
                    "finetuned_from": run_name,
                    "year": yfin,
                    "model": "MultiChannelCNN",
                    "dropout": dropout_p,
                    "downscaling": ds,
                    "normalized": True,
                    "learning_rate": lr,
                    "batch_size": batch_size,
                    "num_epochs": epochs,
                    "station": station,
                    "output_dim": output_dim,
                    "excluded_variables": EXCLUDED_VARS,
                },
                predictions=(ft_metrics.get('predictions') if (isinstance(ft_metrics, dict) and save_predictions) else None),
                output_parameters=ALL_VARS,  # Ajouter les noms des variables
                include_timestamp=True
            )
            
            # Libération mémoire après chaque finetune
            del Xf, yf, ds_finetune, Xf_raw, yf_raw
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
            
        except Exception as e:
            print(f"! Finetune {yfin} échoué: {e}")
            import traceback
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Sweep d'entraînements MultiChannelCNN + finetune années suivantes")
    parser.add_argument("--dataset-dir", type=str, default="dataset", help="Dossier contenant les .h5")
    parser.add_argument("--station", type=str, default="sta69029001", help="ID station (ex: sta69029001)")
    parser.add_argument("--base-year", type=int, default=2016, help="Année de base pour l'entraînement")
    parser.add_argument("--finetune-years", type=int, nargs="*", default=[2017, 2018], help="Années de finetune")
    parser.add_argument("--ds-list", type=int, nargs="*", default=[5, 10], help="Liste des downscalings")
    parser.add_argument("--epochs-list", type=int, nargs="*", default=[10, 20], help="Variations d'époques")
    parser.add_argument("--batch-list", type=int, nargs="*", default=[128, 300], help="Tailles de batch")
    parser.add_argument("--lr-list", type=float, nargs="*", default=[1e-3, 1e-4], help="Learning rates")
    parser.add_argument("--dropout-list", type=float, nargs="*", default=[0.0, 0.05, 0.10, 0.25], help="Dropout en fraction")
    parser.add_argument("--no-cache", action="store_true", help="Désactiver le cache des datasets")
    parser.add_argument("--save-predictions", action="store_true", help="Sauvegarder les prédictions dans les checkpoints")
    parser.add_argument("--no-skip", action="store_true", help="Ne pas skip les modèles déjà entraînés (ré-entraîner)")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)

    # Précharger tous les datasets nécessaires si cache activé
    if not args.no_cache:
        print("\n[PRELOAD] Préchargement des datasets en cache...")
        years_needed = [args.base_year] + args.finetune_years
        for year in years_needed:
            for ds in args.ds_list:
                try:
                    load_dataset_h5(dataset_dir, year, args.station, ds, use_cache=True)
                except Exception as e:
                    print(f"! Impossible de précharger {year} ds{ds}: {e}")

    grid = itertools.product(args.ds_list, args.epochs_list, args.batch_list, args.lr_list, args.dropout_list)
    grid = list(grid)

    print(f"\n{'='*80}")
    print(f"Variables de sortie: {ALL_VARS}")
    print(f"Variables exclues: {EXCLUDED_VARS}")
    print(f"Configurations totales: {len(grid)}")
    print(f"Cache datasets: {'ACTIVÉ' if not args.no_cache else 'DÉSACTIVÉ'}")
    print(f"{'='*80}\n")
    
    for idx, (ds, epochs, batch_size, lr, dropout_p) in enumerate(grid, 1):
        print(f"\n{'='*80}")
        print(f"[{idx}/{len(grid)}] ds={ds}, ep={epochs}, bs={batch_size}, lr={lr}, dropout={dropout_p}")
        print(f"{'='*80}")
        try:
            run_one_training(
                dataset_dir=dataset_dir,
                station=args.station,
                base_year=args.base_year,
                finetune_years=args.finetune_years,
                ds=ds,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                dropout_p=dropout_p,
                save_predictions=args.save_predictions,
                skip_existing=not args.no_skip
            )
        except Exception as e:
            print(f"! ÉCHEC config {idx}/{len(grid)}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()