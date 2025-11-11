import h5py
import numpy as np
from training.train import train_model
from models import *
from torch.utils.data import TensorDataset
import torch
from training.save import save_model_checkpoint

if __name__ == "__main__":
    # Configuration des paramètres
    dataset_path = "dataset/meteonet_SE_2016_sta69029001.h5"
    sample_size = 100

    # Chargement des données h5
    with h5py.File(dataset_path, 'r') as f:
        # Charger les images et labels (limiter au sample_size)
        X = torch.from_numpy(f['images'][:sample_size].astype(np.float32))
        y = torch.from_numpy(f['labels'][:sample_size].astype(np.float32))
        
        # Afficher les informations du dataset
        print(f"Dataset chargé: {dataset_path}")
        print(f"  Images shape: {X.shape}")  # (sample_size, n_timesteps, n_channels, H, W)
        print(f"  Labels shape: {y.shape}")  # (sample_size, n_labels)

        # === DIAGNOSTIC DES NaN ===
        print(f"\n🔍 DIAGNOSTIC DES DONNÉES:")
        print(f"  NaN dans X: {torch.isnan(X).sum().item()} / {X.numel()} ({100*torch.isnan(X).sum().item()/X.numel():.2f}%)")
        print(f"  NaN dans y: {torch.isnan(y).sum().item()} / {y.numel()} ({100*torch.isnan(y).sum().item()/y.numel():.2f}%)")

        # === NETTOYAGE DES NaN ===
        # Option 1: Remplacer les NaN par 0 dans les images
        X = torch.nan_to_num(X, nan=0.0)
        y = torch.nan_to_num(y, nan=0.0)


        
        # Option 2: Supprimer les échantillons avec NaN dans X
        #valid_mask = ~torch.isnan(X).any(dim=1)
        #X = X[valid_mask]
        #y = y[valid_mask]
        
        print(f"\n✅ APRÈS NETTOYAGE:")
        print(f"  Samples restants: {X.shape[0]}/{sample_size}")
        print(f"  NaN dans X: {torch.isnan(X).sum().item()}")
        print(f"  NaN dans y: {torch.isnan(y).sum().item()}")

        # Fusionner les dimensions (n_deltas, n_types) → n_deltas * n_types
        X = X.permute(0, 2, 1, 3, 4)  # (batch, n_types, n_deltas, H, W)
        #X = X.reshape(X.size(0), X.size(1) * X.size(2), X.size(3), X.size(4))  # (batch, 5*4=20, H, W)
        
        # === NORMALISATION DES IMAGES ===
        # Calculer mean/std par canal (sur données valides)
        #X_mean = X.mean(dim=[0, 2, 3], keepdim=True)  # (1, 20, 1, 1)
        #X_std = X.std(dim=[0, 2, 3], keepdim=True)    # (1, 20, 1, 1)
        #X_std = torch.clamp(X_std, min=1e-6)  # éviter division par 0
        #X = (X - X_mean) / X_std
        
        print(f"\n📊 APRÈS NORMALISATION:")
        print(f"  X - min: {X.min():.2f}, max: {X.max():.2f}, mean: {X.mean():.4f}, std: {X.std():.4f}")
        print(f"  Images shape après reshape: {X.shape}")
        
        # Afficher les attributs si disponibles
        if 'channels' in f.attrs:
            print(f"  Canaux: {list(f.attrs['channels'])}")
        if 'target_vars' in f.attrs:
            print(f"  Variables cibles: {list(f.attrs['target_vars'])}")

    dataset = TensorDataset(X, y)

    model = MultiTypeTemporalCorrelationCNN(
        n_types=5,      
        n_deltas=4,     
        output_dim=7    
    )

    params = [
        [10, 100, 1e-1],
        #[20, 100, 1e-1],
        #[50, 100, 1e-1],
        #[100, 100, 1e-1],

    ]
    for p in params:
        trained_model, optimizer, metrics = train_model("MultiChannelTemporalCorrelationCNN-w/o-dropout", model, dataset, num_epochs=p[0], batch_size=p[1], learning_rate=p[2], return_predictions=True)
        
        save_model_checkpoint(
            model=trained_model,
            optimizer=optimizer,
            model_name="MultiChannelTemporalCorrelationCNN",
            epoch=metrics['epoch'],
            train_loss=metrics['train_loss'],
            val_loss=metrics['val_loss'],
            test_loss=metrics['test_loss'],
            hyperparameters=metrics['hyperparameters'],
            predictions=metrics.get('predictions', None),
            include_timestamp=True
        )