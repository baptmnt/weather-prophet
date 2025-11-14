"""
Utilitaires pour sauvegarder les modèles PyTorch
"""
import torch
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import json
import os
import time


def _short_name(hparams: dict, finetune_year: int | None = None, timestamp: str | None = None) -> str:
    def lr_compact(v):
        try:
            return f"lr{v:.0e}".replace('+', '').replace('-', '')
        except Exception:
            return f"lr{v}"
    def year_short(y):
        return f"y{str(y)[-2:]}"
    model = hparams.get("model", "MCNN")
    model = "MCNN" if "MultiChannelCNN" in model else model
    ds = hparams.get("downscaling")
    ds_part = f"d{ds}" if ds is not None else ""
    dropout = hparams.get("dropout", 0)
    do_part = f"do{int(round(float(dropout)*100))}"
    lr = lr_compact(hparams.get("learning_rate", 0))
    bs = hparams.get("batch_size", None)
    b_part = f"b{bs}" if bs is not None else ""
    ep = hparams.get("num_epochs") or hparams.get("epochs")
    e_part = f"e{ep}" if ep is not None else ""
    year = hparams.get("base_year") or hparams.get("year")
    y_part = year_short(year) if year else ""
    excluded = hparams.get("excluded_variables")
    ex_part = ""
    if excluded:
        ex_part = "ex_" + "+".join(excluded)
    parts = [model, ds_part, do_part, lr, b_part, e_part, y_part]
    if ex_part:
        parts.append(ex_part)
    if finetune_year is not None:
        parts.append(f"ft{str(finetune_year)[-2:]}")
    core = "_".join([p for p in parts if p])
    if timestamp:
        core = f"{core}_{timestamp}"
    return core


def save_model_checkpoint(
    model,
    optimizer,
    model_name: str,
    epoch: int,
    train_loss: float = None,
    val_loss: float = None,
    test_loss: float = None,
    hyperparameters: dict = None,
    predictions: dict = None,
    output_parameters: list = None,
    include_timestamp: bool = True
):
    timestamp = time.strftime("%Y%m%d_%H%M%S") if include_timestamp else None
    hparams = hyperparameters or {}
    finetune_year = hparams.get("year") if hparams.get("finetuned_from") else None
    short_base = _short_name(hparams, finetune_year=finetune_year, timestamp=timestamp)
    out_dir = Path("saved_models") / short_base
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fichiers
    json_path = out_dir / f"{short_base}.json"
    weights_path = out_dir / f"{short_base}.pth"

    # Sauvegarde poids
    try:
        torch.save(model.state_dict(), weights_path)
    except Exception as e:
        print(f"[SAVE] Poids non sauvegardés: {e}")

    # Préparer contenu JSON
    payload = {
        "model_name": short_base,
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "test_loss": test_loss,
        "hyperparameters": hparams,
        "save_date": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if output_parameters:
        payload["output_parameters"] = output_parameters
    if predictions:
        payload["predictions"] = predictions

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[SAVE] Checkpoint compact: {json_path}")
    return json_path, weights_path


def save_best_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    model_name: str,
    epoch: int,
    train_loss: float,
    val_loss: float,
    test_loss: float,
    hyperparameters: Dict[str, Any],
    save_dir: str = "saved_models"
) -> Path:
    """
    Sauvegarde le modèle en tant que "best model" (écrase le précédent best).
    Utile pour garder uniquement le meilleur modèle pendant l'entraînement.
    
    Args:
        Même que save_model_checkpoint
    
    Returns:
        Path: Chemin vers le fichier best_model.pth
    """
    save_path = Path(save_dir) / model_name
    save_path.mkdir(parents=True, exist_ok=True)
    
    best_model_path = save_path / "best_model.pth"
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'test_loss': test_loss,
        'hyperparameters': hyperparameters,
        'model_name': model_name,
        'save_date': datetime.now().isoformat()
    }
    
    torch.save(checkpoint, best_model_path)
    
    print(f"🏆 Best model sauvegardé: {best_model_path}")
    print(f"   Test loss: {test_loss:.4f}")
    
    return best_model_path