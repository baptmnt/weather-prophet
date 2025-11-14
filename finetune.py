import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from models import *  # exposes MultiChannelCNN, MultiTypeTemporalCorrelationCNN, etc.
from training.train import train_model
from training.save import save_model_checkpoint
from training.normalisation import (
    normalize_dd, normalize_ff, normalize_precip, normalize_hu,
    normalize_td, normalize_t, normalize_psl,
)

# Ordre canonique des sorties
ALL_VARS = ['dd', 'ff', 'precip', 'hu', 'td', 't', 'psl']
NORM_FUNCS = {
    'dd': normalize_dd,
    'ff': normalize_ff,
    'precip': normalize_precip,
    'hu': normalize_hu,
    'td': normalize_td,
    't': normalize_t,
    'psl': normalize_psl,
}


def find_latest_checkpoint(search_dirs: List[Path], pattern: str = "*.pth") -> Optional[Path]:
    """Trouve le .pth le plus récent dans les dossiers fournis (récursif)."""
    files: List[Path] = []
    for d in search_dirs:
        if d.exists():
            files.extend(d.rglob(pattern))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def find_sibling_json(ckpt_path: Path) -> Optional[Path]:
    """Tente de trouver un JSON à côté du .pth (même prefix/stem)."""
    parent = ckpt_path.parent
    stem = ckpt_path.stem
    # 1) même stem
    cand = parent / f"{stem}.json"
    if cand.exists():
        return cand
    # 2) n'importe quel json le plus récent du dossier
    jsons = list(parent.glob("*.json"))
    if jsons:
        return max(jsons, key=lambda p: p.stat().st_mtime)
    return None


def parse_excluded_from_model_name(model_name: str) -> List[str]:
    """
    Extrait les variables exclues depuis le nom (tokens finaux parmi ALL_VARS).
    Ex: "...-precip-dd" -> ["precip","dd"]
    """
    if not model_name:
        return []
    parts = model_name.split('-')
    excluded: List[str] = []
    # parcourir de la fin vers le début jusqu'à tomber sur un token non variable
    for part in reversed(parts):
        low = part.lower()
        if low in ALL_VARS:
            excluded.append(low)
        else:
            # Stop dès qu'on ne voit plus de variables
            if excluded:
                break
    return list(reversed(excluded))


def build_output_vars(model_name: Optional[str], json_output_params: Optional[List[str]]) -> List[str]:
    """Construit la liste finale de variables de sortie."""
    if json_output_params:
        return json_output_params
    if model_name:
        excluded = set(parse_excluded_from_model_name(model_name))
        return [v for v in ALL_VARS if v not in excluded]
    return ALL_VARS[:]  # fallback


def load_year_dataset(dataset_dir: Path, year: int, station: str, ds: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Charge un fichier HDF5: meteonet_SE_<year>_<station>_ds<ds>.h5
    Retourne X (float32) et y (float32).
    """
    fname = f"meteonet_SE_{year}_{station}_ds{ds}.h5"
    fpath = dataset_dir / fname
    if not fpath.exists():
        raise FileNotFoundError(f"Dataset introuvable: {fpath}")
    with h5py.File(fpath, 'r') as f:
        X = torch.from_numpy(f['images'][:].astype(np.float32))
        y = torch.from_numpy(f['labels'][:].astype(np.float32))
    return X, y


def clean_and_normalize_targets(y: torch.Tensor, output_vars: List[str]) -> torch.Tensor:
    """
    Remplace NaN par moyenne de colonne et applique la normalisation, puis
    ne garde que les colonnes voulues (dans l'ordre).
    """
    # Remplacement NaN colonne par colonne
    y = y.clone()
    col_means = torch.nanmean(torch.where(torch.isnan(y), torch.tensor(0., dtype=y.dtype), y), dim=0)
    for c in range(y.shape[1]):
        mask = torch.isnan(y[:, c])
        y[mask, c] = col_means[c]

    # Appliquer normalisation par colonne
    # Ordre labels attendu: ['dd','ff','precip','hu','td','t','psl']
    for idx, var in enumerate(ALL_VARS):
        func = NORM_FUNCS[var]
        y[:, idx] = func(y[:, idx])

    # Sélectionner uniquement les colonnes de sortie
    indices = [ALL_VARS.index(v) for v in output_vars]
    return y[:, indices]


def prepare_inputs_for_model(X: torch.Tensor, model_name: str) -> torch.Tensor:
    """
    Prépare X selon le modèle:
    - MultiTypeTemporalCorrelationCNN: 5D (B, n_types, n_deltas, H, W)
    - MultiChannelCNN (classique): 4D (B, n_types*n_deltas, H, W)
    Le X brut vient généralement en (B, n_deltas, n_types, H, W).
    """
    # Nettoyage NaN -> 0
    X = torch.nan_to_num(X, nan=0.0)

    # Permuter pour (B, n_types, n_deltas, H, W)
    if X.dim() == 5 and X.shape[1] != X.shape[2]:
        # X: (B, n_deltas, n_types, H, W) -> (B, n_types, n_deltas, H, W)
        X = X.permute(0, 2, 1, 3, 4).contiguous()

    # Z-score par canal (sur (n_types, n_deltas)) pour la version aplatie et 5D
    if "TemporalCorrelation" in model_name:
        # 5D: calcule stats par canal (types,deltas)
        mean = X.mean(dim=[0, 3, 4], keepdim=True)
        std = X.std(dim=[0, 3, 4], keepdim=True).clamp_min(1e-6)
        X = (X - mean) / std
        return X  # (B, n_types, n_deltas, H, W)
    else:
        # Aplatir canaux: (B, n_types*n_deltas, H, W)
        B, T, D, H, W = X.shape
        X = X.view(B, T * D, H, W).contiguous()
        mean = X.mean(dim=[0, 2, 3], keepdim=True)
        std = X.std(dim=[0, 2, 3], keepdim=True).clamp_min(1e-6)
        X = (X - mean) / std
        return X  # (B, C, H, W)


def build_model_from_checkpoint_meta(model_name: str, X: torch.Tensor, output_dim: int):
    """
    Instancie le modèle d'après son nom de base.
    - Si le nom contient 'MultiTypeTemporalCorrelationCNN' → modèle temporel 5D
    - Sinon, fallback MultiChannelCNN 2D
    """
    # Détermination n_types/n_deltas
    if X.dim() == 5:
        n_types, n_deltas = X.shape[1], X.shape[2]
    elif X.dim() == 4:
        # Impossible de déduire directement; essayer de deviner via racine carrée?
        # On ne va pas s'appuyer dessus; on utilisera l'original avant aplatissement.
        raise ValueError("X 4D reçu. Fournir X 5D à build_model_from_checkpoint_meta.")

    if "MultiTypeTemporalCorrelationCNN" in model_name or "TemporalCorrelation" in model_name:
        return MultiTypeTemporalCorrelationCNN(
            n_types=n_types,
            n_deltas=n_deltas,
            output_dim=output_dim
        )
    else:
        return MultiChannelCNN(
            n_types=n_types,
            n_deltas=n_deltas,
            output_dim=output_dim
        )


def robust_load_state_dict(model: torch.nn.Module, ckpt: Dict) -> None:
    """Charge state_dict de manière robuste depuis divers formats de .pth."""
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        sd = ckpt['model_state_dict']
    elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
        sd = ckpt['state_dict']
    else:
        sd = ckpt
    model.load_state_dict(sd, strict=False)


def main():
    parser = argparse.ArgumentParser(description="Finetune sur l'année suivante à partir d'un .pth existant")
    parser.add_argument('--year', type=int, required=True, help='Année cible pour le finetune (ex: 2017)')
    parser.add_argument('--prev-year', type=int, default=None, help="Année d'origine du pré-entraînement (optionnel, informatif)")
    parser.add_argument('--station', type=str, default="sta69029001", help="Identifiant station (ex: sta69029001)")
    parser.add_argument('--ds', type=int, default=5, choices=[2, 5, 10], help="Downscaling (2/5/10)")
    parser.add_argument('--dataset-dir', type=str, default="dataset", help="Dossier des datasets .h5")
    parser.add_argument('--ckpt', type=str, default=None, help="Chemin explicite vers le .pth à charger")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--batch-size', type=int, default=300, help="Batch size")
    parser.add_argument('--epochs', type=int, default=10, help="Nombre d’époques pour le finetune")
    parser.add_argument('--save-name', type=str, default=None, help="Nom explicite du modèle sauvegardé")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)

    # 1) Trouver le checkpoint .pth
    if args.ckpt:
        ckpt_path = Path(args.ckpt)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint non trouvé: {ckpt_path}")
    else:
        ckpt_path = find_latest_checkpoint([Path("saved_models"), Path("runs")])
        if ckpt_path is None:
            raise FileNotFoundError("Aucun .pth trouvé sous saved_models/ ou runs/")
    print(f"[+] Checkpoint: {ckpt_path}")

    # 2) Charger JSON associé (métadonnées)
    json_path = find_sibling_json(ckpt_path)
    model_name = ckpt_path.stem
    saved_hparams: Optional[Dict] = None
    json_output_params: Optional[List[str]] = None
    if json_path:
        try:
            meta = json.loads(json_path.read_text(encoding="utf-8"))
            # Deux formats possibles
            model_name = meta.get('model_name', model_name)
            saved_hparams = meta.get('hyperparameters') or meta.get('params') or None
            json_output_params = meta.get('output_parameters') or None
        except Exception:
            pass
    print(f"[+] Model name: {model_name}")

    # 3) Charger dataset cible (année suivante)
    X_raw, y_raw = load_year_dataset(dataset_dir, args.year, args.station, args.ds)
    print(f"[+] Dataset {args.year}: X={tuple(X_raw.shape)}, y={tuple(y_raw.shape)}")

    # 4) Déterminer les variables de sortie
    output_vars = build_output_vars(model_name, json_output_params)
    output_dim = len(output_vars)
    print(f"[+] Sorties: {output_vars} (dim={output_dim})")

    # 5) Préparer X (5D pour déduction n_types/n_deltas et normalisation)
    # X_raw: attendu (B, n_deltas, n_types, H, W)
    if X_raw.dim() != 5:
        raise ValueError(f"Format X inattendu, attendu 5D (B, n_deltas, n_types, H, W), reçu {tuple(X_raw.shape)}")
    # Préparer inputs selon le modèle (on utilisera model_name pour décider)
    # On garde une version 5D (B,T,D,H,W) pour construire le modèle
    X_5d = X_raw.permute(0, 2, 1, 3, 4).contiguous()  # (B, n_types, n_deltas, H, W)

    # 6) Construire le modèle (selon model_name)
    model = build_model_from_checkpoint_meta(model_name, X_5d, output_dim=output_dim)

    # 7) Charger le state_dict depuis le .pth
    ckpt_blob = torch.load(ckpt_path, map_location="cpu")
    robust_load_state_dict(model, ckpt_blob)

    # 8) Finaliser les entrées pour l'entraînement (normalisation images)
    if isinstance(model, MultiTypeTemporalCorrelationCNN):
        X = prepare_inputs_for_model(X_raw, "TemporalCorrelation")  # 5D normalisé
    else:
        X = prepare_inputs_for_model(X_raw, "MultiChannelCNN")      # 4D normalisé

    # 9) Nettoyer + normaliser y, puis sélectionner colonnes
    y = clean_and_normalize_targets(y_raw, output_vars)

    # 10) Dataset + entraînement
    dataset = TensorDataset(X, y)

    # Nom de sauvegarde
    ds_size = len(dataset)
    save_name = args.save_name or f"{model_name}_finetune{args.year}_ds{ds_size}_lr{args.lr}_bs{args.batch_size}_ep{args.epochs}"

    trained_model, optimizer, metrics = train_model(
        name=save_name,
        model=model,
        dataset=dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        return_predictions=True
    )

    # 11) Sauvegarde
    try:
        save_model_checkpoint(
            model=trained_model,
            optimizer=optimizer,
            model_name=save_name,
            epoch=metrics.get('epoch', args.epochs),
            train_loss=metrics.get('train_loss'),
            val_loss=metrics.get('val_loss'),
            test_loss=metrics.get('test_loss'),
            hyperparameters=metrics.get('hyperparameters', {
                "finetuned_from": str(ckpt_path),
                "year": args.year,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "num_epochs": args.epochs,
                "output_parameters": output_vars,
            }),
            predictions=metrics.get('predictions'),
            include_timestamp=True
        )
    except Exception as e:
        print(f"[!] Erreur sauvegarde checkpoint: {e}")

    print("[✓] Finetune terminé.")


if __name__ == "__main__":
    main()