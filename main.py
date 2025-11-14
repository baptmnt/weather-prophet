from training.train import train_model
from models import *
from torch.utils.data import TensorDataset
from training.save import save_model_checkpoint
from training.normalisation import normalize_dd, normalize_ff, normalize_precip, normalize_hu, normalize_td, normalize_t, normalize_psl
import torch
import h5py
import numpy as np

if __name__ == "__main__":
    # Configuration des paramètres
    dataset_path = "dataset/meteonet_SE_2016_sta69029001_ds5.h5"
    sample_size = 8737

    # Chargement des données h5
    with h5py.File(dataset_path, 'r') as f:
        # Charger les images et labels
        X = torch.from_numpy(f['images'][:sample_size].astype(np.float32))
        y = torch.from_numpy(f['labels'][:sample_size].astype(np.float32))
        
        print(f"Dataset chargé: {dataset_path}")
        

        # Diagnostic et nettoyage des NaN
        print(f"\n🔍 DIAGNOSTIC DES DONNÉES:")
        print(f"  NaN dans X: {torch.isnan(X).sum().item()} / {X.numel()}")
        print(f"  NaN dans y: {torch.isnan(y).sum().item()} / {y.numel()}")

        # Nettoyage X
        X = torch.nan_to_num(X, nan=0.0)
        y = torch.nan_to_num(y, nan=0.0)


        
        # Nettoyage y (garder lignes avec au moins 3 valeurs valides)
        # valid_mask = ~torch.isnan(y)
        # X = X[valid_mask]
        # y = y[valid_mask]
        
        # Remplacer NaN restants par la moyenne de colonne
        # y_mean = torch.nanmean(y, dim=0)
        # for col_idx in range(y.shape[1]):
        #     col_mask = torch.isnan(y[:, col_idx])
        #     y[col_mask, col_idx] = y_mean[col_idx]
        
        print(f"✅ Samples après nettoyage: {X.shape[0]}")

        # Normalisation des targets (y)
        # Ordre: ['dd', 'ff', 'precip', 'hu', 'td', 't', 'psl']
        print(f"\n📐 NORMALISATION DES TARGETS:")
        y[:, 0] = normalize_dd(y[:, 0])      # dd (direction vent)
        # max sur l'axe :, 1
        y[:, 1] = normalize_ff(y[:, 1])      # ff (vitesse vent)
        y[:, 2] = normalize_precip(y[:, 2])  # precip
        y[:, 3] = normalize_hu(y[:, 3])      # hu (humidité)
        y[:, 4] = normalize_td(y[:, 4])      # td (point de rosée)
        y[:, 5] = normalize_t(y[:, 5])       # t (température)
        y[:, 6] = normalize_psl(y[:, 6])     # psl (pression)

        # Supprimer l'axe 0 et 2 : dd et precip
        y = torch.cat([y[:, 1:2], y[:, 3:]], dim=1)

        print(f"  Images shape: {X.shape}")
        print(f"  Labels shape: {y.shape}")



        print(f"  ✓ Targets normalisés")
        print(f"  Range après normalisation: [{y.min():.3f}, {y.max():.3f}]")

        # Reshape et normalisation des images (X)
        X = X.permute(0, 2, 1, 3, 4)
        X = X.reshape(X.size(0), X.size(1) * X.size(2), X.size(3), X.size(4))
        
        # X_mean = X.mean(dim=[0, 2, 3], keepdim=True)
        # X_std = X.std(dim=[0, 2, 3], keepdim=True)
        # X_std = torch.clamp(X_std, min=1e-6)
        # X = (X - X_mean) / X_std
        
        print(f"  Images shape après reshape: {X.shape}")

    dataset = TensorDataset(X, y)

    # Hyperparamètres à tester
    params = [
        (50, 300, 1e-3),
        (50, 300, 1e-4),

    ]
    
    for i, p in enumerate(params):
        print(f"\n{'='*70}")
        print(f"ENTRAÎNEMENT {i+1}/{len(params)}")
        print(f"{'='*70}")
        
        # Créer un nouveau modèle pour chaque config
        model = MultiChannelCNN(
            n_types=5,      
            n_deltas=4,     
            output_dim=5
        )
        
        # Entraîner le modèle
        trained_model, optimizer, metrics = train_model(
            name="MultiChannelCNN-wo-dropout-ds5-normalized-R2-precip-dd",
            model=model,
            dataset=dataset,
            num_epochs=p[0],
            batch_size=p[1],
            learning_rate=p[2],
            return_predictions=True
        )
        
        # Sauvegarder ce modèle
        preds = metrics.get("predictions", {}).get("predictions")
        trues = metrics.get("predictions", {}).get("targets")
        target_vars = ['ff', 'hu', 'td', 't', 'psl']
        
        save_model_checkpoint(
            model=trained_model,
            optimizer=optimizer,
            model_name="MultiChannelCNN-wo-dropout-ds5-normalized-R2-precip-dd",
            epoch=metrics['epoch'],
            train_loss=metrics['train_loss'],
            val_loss=metrics['val_loss'],
            test_loss=metrics['test_loss'],
            hyperparameters=metrics['hyperparameters'],
            predictions=metrics.get('predictions', None),
            include_timestamp=True,
            output_parameters=target_vars
        )