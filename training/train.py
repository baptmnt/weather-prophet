# Le but de ce fichier est d'offrir une function permettant d'entrainer
# un modèle, en faisant varier des paramètres (epoch, batch size, learning rate, etc.)
# Tout en exportant les résultats dans TensorBoard

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from typing import Dict, Tuple, List
import numpy as np

def mae(pred, target):
    return torch.mean(torch.abs(pred - target))

def r2_score(pred, target):
    ss_res = torch.sum((target - pred) ** 2)
    ss_tot = torch.sum((target - torch.mean(target, dim=0)) ** 2)
    return 1 - ss_res / ss_tot


def train_model(
    name: str, 
    model: nn.Module, 
    dataset, 
    train_size=0.8, 
    val_size=0.1, 
    num_epochs=10, 
    batch_size=32, 
    learning_rate=1e-3,
    return_predictions=False  # Nouveau paramètre
) -> Tuple[nn.Module, torch.optim.Optimizer, Dict]:
    """
    Entraîne un modèle PyTorch.
    
    Args:
        name: Nom du modèle pour TensorBoard
        model: Modèle PyTorch à entraîner
        dataset: Dataset PyTorch
        train_size: Proportion du dataset pour l'entraînement
        val_size: Proportion du dataset pour la validation
        num_epochs: Nombre d'époques
        batch_size: Taille des batchs
        learning_rate: Taux d'apprentissage
        return_predictions: Si True, retourne les prédictions sur le test set
    
    Returns:
        model: Modèle entraîné
        optimizer: Optimizer (pour sauvegarder son état)
        metrics: Dict contenant toutes les métriques d'entraînement
    """
    # Division du dataset en train, val, test
    train_size_count = int(train_size * len(dataset))
    val_size_count = int(val_size * len(dataset))
    test_size_count = len(dataset) - train_size_count - val_size_count

    train_set, val_set, test_set = random_split(dataset, [train_size_count, val_size_count, test_size_count])

    # Création des DataLoaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    writer = SummaryWriter(log_dir=f"runs/{name}_ds{len(dataset)}_lr{learning_rate}_bs{batch_size}_ep{num_epochs}")

    # Gradient clipping
    max_grad_norm = 1.0
    
    # Tracking des métriques
    final_train_loss = 0.0
    final_val_loss = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_mae, train_r2 = 0.0, 0.0, 0.0

        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Vérifier les NaN avant la loss
            if torch.isnan(outputs).any():
                print(f"⚠️ NaN détecté dans les outputs à l'époque {epoch+1}")
                continue
            
            loss = criterion(outputs, targets)
            
            if torch.isnan(loss):
                print(f"⚠️ NaN détecté dans la loss à l'époque {epoch+1}")
                continue
            
            loss.backward()
            
            # Gradient clipping pour éviter l'explosion
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_mae  += mae(outputs, targets).item() * inputs.size(0)
            train_r2   += r2_score(outputs, targets).item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)
        train_mae  /= len(train_loader.dataset)
        train_r2   /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss, val_mae, val_r2 = 0.0, 0.0, 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                val_mae  += mae(outputs, targets).item() * inputs.size(0)
                val_r2   += r2_score(outputs, targets).item() * inputs.size(0)

        val_loss /= len(val_loader.dataset)
        val_mae  /= len(val_loader.dataset)
        val_r2   /= len(val_loader.dataset)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Metric/train_MAE", train_mae, epoch)
        writer.add_scalar("Metric/val_MAE", val_mae, epoch)
        writer.add_scalar("Metric/train_R2", train_r2, epoch)
        writer.add_scalar("Metric/val_R2", val_r2, epoch)

        print(f"Époque [{epoch+1}/{num_epochs}]  |  Perte train: {train_loss:.4f}  |  Perte val: {val_loss:.4f}")
        
        # Sauvegarder les dernières métriques
        final_train_loss = train_loss
        final_val_loss = val_loss
    
    writer.close()

    # Test final avec collecte des prédictions
    model.eval()
    test_loss, test_mae, test_r2 = 0.0, 0.0, 0.0
    
    # Listes pour stocker les prédictions et vraies valeurs
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)
            test_mae  += mae(outputs, targets).item() * inputs.size(0)
            test_r2   += r2_score(outputs, targets).item() * inputs.size(0)
            
            # Collecter les prédictions et vraies valeurs
            if return_predictions and len(all_predictions) < 10:  # Limiter pour éviter trop de mémoire
                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
    
    test_loss /= len(test_loader.dataset)
    test_mae  /= len(test_loader.dataset)
    test_r2   /= len(test_loader.dataset)
    
    print(f"\n{'='*70}")
    print(f"RÉSULTATS FINAUX")
    print(f"{'='*70}")
    print(f"Train loss: {final_train_loss:.4f}")
    print(f"Val loss:   {final_val_loss:.4f}")
    print(f"Test loss:  {test_loss:.4f}")
    print(f"Test MAE:   {test_mae:.4f}")
    print(f"Test R²:    {test_r2:.4f}")
    print(f"{'='*70}\n")

    # Concaténer toutes les prédictions
    predictions_dict = None
    if return_predictions:
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        #Garder que quelques échantillons pour éviter trop de mémoire
        predictions_dict = {
            'predictions':all_predictions,  # Shape: (n_test_samples, output_dim)
            'targets': all_targets,          # Shape: (n_test_samples, output_dim)
            'residuals': all_predictions - all_targets,  # Erreurs
        }
        #Passer d'un type numpy à json serializable
        for key in predictions_dict:
            predictions_dict[key] = predictions_dict[key].tolist()
        

        
        print(f"📊 Prédictions collectées:")
        print(f"  Shape predictions: {all_predictions.shape}")
        print(f"  Shape targets: {all_targets.shape}")

    # Préparer les métriques à retourner
    metrics = {
        'epoch': num_epochs,
        'train_loss': final_train_loss,
        'val_loss': final_val_loss,
        'test_loss': test_loss,
        'test_mae': test_mae,
        'test_r2': test_r2,
        'hyperparameters': {
            'n_types': getattr(model, 'n_types', 5),
            'n_deltas': getattr(model, 'n_deltas', 4),
            'output_dim': getattr(model, 'output_dim', 7),
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'dataset_size': len(dataset)
        },
        'predictions': predictions_dict  # Ajouter les prédictions aux métriques
    }
    print(predictions_dict)
    return model, optimizer, metrics