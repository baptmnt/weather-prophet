"""
Exemple d'utilisation du dataset HDF5 avec PyTorch.

Définit un DataLoader PyTorch pour charger les données efficacement.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


class MeteoNetDataset(Dataset):
    """
    Dataset PyTorch pour le dataset MeteoNet HDF5.
    
    Structure des données:
        - Images: (n_timesteps, n_channels, height, width)
        - Labels: (n_labels,) 
        - Metadata: timestamp, station_id, coordinates
    
    Args:
        h5_path: Chemin vers le fichier HDF5
        transform: Transformations optionnelles à appliquer aux images
        target_transform: Transformations optionnelles à appliquer aux labels
        normalize: Si True, normalise les images par canal
        handle_nans: Comment gérer les NaN - 'zero', 'mean', ou 'keep'
    """
    
    def __init__(
        self,
        h5_path: Path,
        transform=None,
        target_transform=None,
        normalize: bool = True,
        handle_nans: str = 'zero'
    ):
        self.h5_path = h5_path
        self.transform = transform
        self.target_transform = target_transform
        self.normalize = normalize
        self.handle_nans = handle_nans
        
        # Ouvrir le fichier en mode lecture pour accéder aux attributs
        with h5py.File(self.h5_path, 'r') as f:
            self.n_samples = f.attrs['n_samples']
            self.n_timesteps = f.attrs['n_timesteps']
            self.n_channels = f.attrs['n_channels']
            self.image_height = f.attrs['image_height']
            self.image_width = f.attrs['image_width']
            self.channels = list(f.attrs['channels'])
            self.target_vars = list(f.attrs['target_vars'])
            self.timesteps = list(f.attrs['timesteps'])
            
            # Calculer les statistiques pour la normalisation si nécessaire
            if self.normalize:
                self._compute_normalization_stats()
    
    def _compute_normalization_stats(self):
        """Calcule mean et std pour chaque canal (sur un échantillon)"""
        print("Calcul des statistiques de normalisation...")
        
        with h5py.File(self.h5_path, 'r') as f:
            # Échantillonner pour calculer les stats (pas tout le dataset)
            n_samples_for_stats = min(500, self.n_samples)
            indices = np.random.choice(self.n_samples, n_samples_for_stats, replace=False)
            indices = np.sort(indices)
            
            sample_images = f['images'][indices]
            
            # Calculer mean et std par canal
            self.channel_mean = []
            self.channel_std = []
            
            for c_idx in range(self.n_channels):
                channel_data = sample_images[:, :, c_idx, :, :]
                valid_data = channel_data[np.isfinite(channel_data)]
                
                if len(valid_data) > 0:
                    self.channel_mean.append(float(valid_data.mean()))
                    self.channel_std.append(float(valid_data.std()))
                else:
                    self.channel_mean.append(0.0)
                    self.channel_std.append(1.0)
            
            print(f"  Mean par canal: {[f'{m:.2f}' for m in self.channel_mean]}")
            print(f"  Std par canal: {[f'{s:.2f}' for s in self.channel_std]}")
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Retourne un sample du dataset.
        
        Returns:
            images: Tensor de shape (n_timesteps, n_channels, height, width)
            labels: Tensor de shape (n_labels,)
            metadata: Dict avec timestamp, station_id, coords
        """
        # Ouvrir le fichier HDF5 (chaque worker aura sa propre connexion)
        with h5py.File(self.h5_path, 'r') as f:
            # Charger les images
            images = f['images'][idx].astype(np.float32)
            
            # Charger les labels
            labels = f['labels'][idx].astype(np.float32)
            
            # Charger metadata
            metadata = {
                'timestamp': f['metadata/timestamps'][idx].decode('utf-8'),
                'station_id': int(f['metadata/station_ids'][idx]),
                'station_coords': f['metadata/station_coords'][idx].astype(np.float32),
                'station_height': float(f['metadata/station_heights'][idx]),
                'zone': f['metadata/zones'][idx].decode('utf-8'),
            }
        
        # Gérer les NaN dans les images
        if self.handle_nans == 'zero':
            images = np.nan_to_num(images, nan=0.0)
        elif self.handle_nans == 'mean':
            for c_idx in range(self.n_channels):
                channel_data = images[:, c_idx, :, :]
                if self.normalize:
                    fill_value = self.channel_mean[c_idx]
                else:
                    fill_value = np.nanmean(channel_data)
                    if np.isnan(fill_value):
                        fill_value = 0.0
                channel_data[np.isnan(channel_data)] = fill_value
                images[:, c_idx, :, :] = channel_data
        # Si 'keep', on garde les NaN (pas recommandé pour l'entraînement)
        
        # Normaliser si demandé
        if self.normalize:
            for c_idx in range(self.n_channels):
                images[:, c_idx, :, :] = (images[:, c_idx, :, :] - self.channel_mean[c_idx]) / (self.channel_std[c_idx] + 1e-8)
        
        # Convertir en tensors PyTorch
        images_tensor = torch.from_numpy(images)
        labels_tensor = torch.from_numpy(labels)
        
        # Appliquer les transformations si fournies
        if self.transform:
            images_tensor = self.transform(images_tensor)
        
        if self.target_transform:
            labels_tensor = self.target_transform(labels_tensor)
        
        return images_tensor, labels_tensor, metadata
    
    def get_channel_names(self):
        """Retourne les noms des canaux"""
        return self.channels
    
    def get_target_names(self):
        """Retourne les noms des variables cibles"""
        return self.target_vars
    
    def get_timesteps(self):
        """Retourne les timesteps utilisés"""
        return self.timesteps


def create_dataloaders(
    dataset_path: Path,
    batch_size: int = 32,
    train_split: float = 0.7,
    val_split: float = 0.15,
    num_workers: int = 4,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Crée les DataLoaders train/val/test à partir du dataset HDF5.
    
    Args:
        dataset_path: Chemin vers le fichier HDF5
        batch_size: Taille des batchs
        train_split: Proportion du dataset pour l'entraînement
        val_split: Proportion du dataset pour la validation
        num_workers: Nombre de workers pour le chargement parallèle
        seed: Seed pour la reproductibilité
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Créer le dataset complet
    full_dataset = MeteoNetDataset(dataset_path, normalize=True, handle_nans='zero')
    
    # Calculer les tailles des splits
    dataset_size = len(full_dataset)
    train_size = int(train_split * dataset_size)
    val_size = int(val_split * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    print(f"\nSplits du dataset:")
    print(f"  Train: {train_size} samples ({100*train_split:.0f}%)")
    print(f"  Validation: {val_size} samples ({100*val_split:.0f}%)")
    print(f"  Test: {test_size} samples ({100*(1-train_split-val_split):.0f}%)")
    
    # Split aléatoire
    torch.manual_seed(seed)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    # Créer les DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def demo_usage():
    """Démo d'utilisation du DataLoader"""
    
    # Chemin du dataset
    dataset_path = Path(r"d:\Documents\Scolarité\5 - INSA Lyon\4TCA\S3\TIP\Projet\weather-prophet\datasets\meteonet_SE_2016_20160101.h5")
    
    if not dataset_path.exists():
        print(f"❌ Dataset non trouvé: {dataset_path}")
        return
    
    print(f"📂 Chargement du dataset: {dataset_path.name}")
    
    # Créer les DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_path,
        batch_size=16,
        train_split=0.7,
        val_split=0.15,
        num_workers=0  # 0 pour debug, augmenter pour production
    )
    
    # Tester le chargement d'un batch
    print("\n" + "="*70)
    print("TEST DE CHARGEMENT D'UN BATCH")
    print("="*70)
    
    # Obtenir un batch du train loader
    images, labels, metadata = next(iter(train_loader))
    
    print(f"\nBatch d'entraînement:")
    print(f"  Images shape: {images.shape}")  # (batch, timesteps, channels, h, w)
    print(f"  Labels shape: {labels.shape}")  # (batch, n_labels)
    print(f"  Metadata keys: {metadata.keys()}")
    
    print(f"\nStatistiques du batch:")
    print(f"  Images - min: {images.min():.2f}, max: {images.max():.2f}, mean: {images.mean():.2f}")
    
    # Filtrer les NaN pour calculer min/max des labels
    labels_valid = labels[~torch.isnan(labels)]
    if len(labels_valid) > 0:
        print(f"  Labels - min: {labels_valid.min():.2f}, max: {labels_valid.max():.2f}, mean: {labels_valid.mean():.2f}")
    else:
        print(f"  Labels - Toutes les valeurs sont NaN")
    
    print(f"\nExemple de metadata du premier sample:")
    print(f"  Timestamp: {metadata['timestamp'][0]}")
    print(f"  Station ID: {metadata['station_id'][0]}")
    print(f"  Coordinates: lat={metadata['station_coords'][0][0]:.2f}, lon={metadata['station_coords'][0][1]:.2f}")
    print(f"  Zone: {metadata['zone'][0]}")
    
    # Test d'itération complète sur quelques batchs
    print(f"\n" + "="*70)
    print("TEST D'ITÉRATION")
    print("="*70)
    
    for i, (images, labels, metadata) in enumerate(train_loader):
        if i >= 3:  # Juste 3 batchs pour le test
            break
        print(f"  Batch {i+1}: {images.shape[0]} samples chargés")
    
    print("\n✅ DataLoader fonctionne correctement!")
    
    # Informations pour utilisation dans un modèle
    dataset = train_loader.dataset.dataset  # Accéder au dataset original
    print(f"\n" + "="*70)
    print("INFORMATIONS POUR L'ENTRAÎNEMENT")
    print("="*70)
    print(f"\nNombre de canaux: {dataset.n_channels}")
    print(f"Canaux: {dataset.get_channel_names()}")
    print(f"Timesteps: {dataset.get_timesteps()}")
    print(f"Taille images: {dataset.image_height}×{dataset.image_width}")
    print(f"\nVariables cibles ({len(dataset.get_target_names())}):")
    for i, var in enumerate(dataset.get_target_names()):
        print(f"  {i}: {var}")


if __name__ == "__main__":
    demo_usage()
