"""
Utilitaires pour sauvegarder les modèles PyTorch
"""
import torch
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import json


def save_model_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    model_name: str,
    epoch: int,
    train_loss: float,
    val_loss: float,
    test_loss: float,
    hyperparameters: Dict[str, Any],
    save_dir: str = "saved_models",
    include_timestamp: bool = True,
    predictions: Optional[Any] = None
) -> Path:
    """
    Sauvegarde un checkpoint de modèle avec toutes les informations nécessaires.
    
    Args:
        model: Modèle PyTorch à sauvegarder
        optimizer: Optimizer utilisé pour l'entraînement
        model_name: Nom du modèle (ex: "MultiChannelCNN")
        epoch: Nombre d'époques d'entraînement
        train_loss: Loss finale sur le train set
        val_loss: Loss finale sur le validation set
        test_loss: Loss finale sur le test set
        hyperparameters: Dict contenant les hyperparamètres du modèle
        save_dir: Dossier racine de sauvegarde
        include_timestamp: Ajouter un timestamp au nom du fichier
        predictions: Prédictions du modèle à sauvegarder (optionnel)
    
    Returns:
        Path: Chemin vers le fichier sauvegardé
    
    Example:
        >>> save_model_checkpoint(
        ...     model=my_model,
        ...     optimizer=my_optimizer,
        ...     model_name="MultiChannelCNN",
        ...     epoch=100,
        ...     train_loss=0.123,
        ...     val_loss=0.145,
        ...     test_loss=0.150,
        ...     hyperparameters={
        ...         'n_types': 5,
        ...         'n_deltas': 4,
        ...         'output_dim': 7,
        ...         'learning_rate': 1e-3,
        ...         'batch_size': 32
        ...     }
        ... )
    """
    # Créer le dossier de sauvegarde
    save_path = Path(save_dir) / model_name
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Construire le nom du fichier
    filename_parts = [model_name]
    filename_parts.append(f"ep{epoch}")
    filename_parts.append(f"loss{test_loss:.4f}")
    
    if include_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_parts.append(timestamp)
    
    filename = "_".join(filename_parts) + ".pth"
    checkpoint_path = save_path / filename
    
    # Préparer le checkpoint
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
    
    # Sauvegarder le checkpoint
    torch.save(checkpoint, checkpoint_path)
    
    # Sauvegarder aussi les hyperparamètres en JSON pour référence
    json_path = save_path / f"{filename.replace('.pth', '.json')}"
    with open(json_path, 'w') as f:
        json.dump({
            'model_name': model_name,
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'test_loss': test_loss,
            'hyperparameters': hyperparameters,
            'save_date': datetime.now().isoformat(),
            'checkpoint_file': str(checkpoint_path), 
            'predictions': predictions,
        }, f, indent=2)
    
    print(f"✅ Modèle sauvegardé:")
    print(f"   Checkpoint: {checkpoint_path}")
    print(f"   Config JSON: {json_path}")
    print(f"   Test loss: {test_loss:.4f}")
    
    return checkpoint_path


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