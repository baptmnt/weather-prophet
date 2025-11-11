"""
Script d'inspection et de visualisation du dataset HDF5.

Usage:
    python inspect_dataset.py [chemin_vers_dataset.h5]
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def inspect_hdf5_structure(filepath: Path):
    """Affiche la structure complète du fichier HDF5"""
    print(f"\n{'='*70}")
    print(f"INSPECTION DU DATASET: {filepath.name}")
    print(f"{'='*70}\n")
    
    with h5py.File(filepath, 'r') as f:
        # Attributs globaux
        print("📋 ATTRIBUTS GLOBAUX:")
        for key, value in f.attrs.items():
            print(f"  {key}: {value}")
        
        print("\n📁 STRUCTURE DU FICHIER:")
        
        # Datasets principaux
        print("\n  Images:")
        images = f['images']
        print(f"    Shape: {images.shape}")
        print(f"    Dtype: {images.dtype}")
        print(f"    Size: {images.nbytes / (1024**2):.1f} MB")
        
        print("\n  Labels:")
        labels = f['labels']
        print(f"    Shape: {labels.shape}")
        print(f"    Dtype: {labels.dtype}")
        print(f"    Variables: {f.attrs['target_vars']}")
        
        print("\n  Metadata:")
        metadata = f['metadata']
        for key in metadata.keys():
            dset = metadata[key]
            print(f"    {key}: shape={dset.shape}, dtype={dset.dtype}")
        
        # Statistiques sur les données
        print("\n📊 STATISTIQUES SUR LES LABELS:")
        labels_data = labels[:]
        target_vars = f.attrs['target_vars']
        
        for i, var in enumerate(target_vars):
            var_data = labels_data[:, i]
            valid_data = var_data[np.isfinite(var_data)]
            
            if len(valid_data) > 0:
                print(f"  {var}:")
                print(f"    Min: {valid_data.min():.2f}")
                print(f"    Max: {valid_data.max():.2f}")
                print(f"    Mean: {valid_data.mean():.2f}")
                print(f"    Std: {valid_data.std():.2f}")
                print(f"    NaN count: {np.isnan(var_data).sum()} ({100*np.isnan(var_data).sum()/len(var_data):.1f}%)")
            else:
                print(f"  {var}: Toutes les valeurs sont NaN")
        
        # Statistiques sur les images
        print("\n📊 STATISTIQUES SUR LES IMAGES:")
        
        # Échantillonner quelques images pour les stats (pour ne pas tout charger)
        sample_indices = np.random.choice(len(images), min(100, len(images)), replace=False)
        sample_indices = np.sort(sample_indices)  # HDF5 nécessite des indices triés
        sample_images = images[sample_indices]
        
        channels = f.attrs['channels']
        for c_idx, channel in enumerate(channels):
            channel_data = sample_images[:, :, c_idx, :, :]
            valid_data = channel_data[np.isfinite(channel_data)]
            
            if len(valid_data) > 0:
                print(f"  {channel}:")
                print(f"    Min: {valid_data.min():.2f}")
                print(f"    Max: {valid_data.max():.2f}")
                print(f"    Mean: {valid_data.mean():.2f}")
                print(f"    Std: {valid_data.std():.2f}")
                print(f"    NaN ratio: {100*np.isnan(channel_data).sum()/channel_data.size:.1f}%")
        
        # Info sur les stations
        print("\n📍 STATISTIQUES SPATIALES:")
        coords = metadata['station_coords'][:]
        station_ids = metadata['station_ids'][:]
        
        print(f"  Nombre de samples: {len(station_ids)}")
        print(f"  Stations uniques: {len(np.unique(station_ids))}")
        print(f"  Latitude: min={coords[:, 0].min():.2f}, max={coords[:, 0].max():.2f}")
        print(f"  Longitude: min={coords[:, 1].min():.2f}, max={coords[:, 1].max():.2f}")
        
        # Info temporelle
        print("\n⏰ STATISTIQUES TEMPORELLES:")
        timestamps = metadata['timestamps'][:]
        print(f"  Nombre de timestamps: {len(timestamps)}")
        print(f"  Premier: {timestamps[0].decode('utf-8')}")
        print(f"  Dernier: {timestamps[-1].decode('utf-8')}")


def visualize_sample(filepath: Path, sample_idx: int = 0):
    """Visualise un sample du dataset"""
    
    with h5py.File(filepath, 'r') as f:
        # Charger un sample
        sample_images = f['images'][sample_idx]  # (n_timesteps, n_channels, h, w)
        sample_labels = f['labels'][sample_idx]
        sample_timestamp = f['metadata/timestamps'][sample_idx].decode('utf-8')
        sample_station = f['metadata/station_ids'][sample_idx]
        sample_coords = f['metadata/station_coords'][sample_idx]
        
        channels = f.attrs['channels']
        timesteps = f.attrs['timesteps']
        target_vars = f.attrs['target_vars']
        
        print(f"\n{'='*70}")
        print(f"VISUALISATION DU SAMPLE #{sample_idx}")
        print(f"{'='*70}")
        print(f"  Timestamp: {sample_timestamp}")
        print(f"  Station ID: {sample_station}")
        print(f"  Coordonnées: lat={sample_coords[0]:.2f}, lon={sample_coords[1]:.2f}")
        print(f"\n  Labels:")
        for i, var in enumerate(target_vars):
            value = sample_labels[i]
            if np.isfinite(value):
                print(f"    {var}: {value:.2f}")
            else:
                print(f"    {var}: NaN")
        
        # Créer une figure avec toutes les images
        n_timesteps = len(timesteps)
        n_channels = len(channels)
        
        fig, axes = plt.subplots(n_timesteps, n_channels, figsize=(15, 12))
        fig.suptitle(f'Sample #{sample_idx} - Station {sample_station} - {sample_timestamp}', 
                     fontsize=14, fontweight='bold')
        
        for t_idx, timestep in enumerate(timesteps):
            for c_idx, channel in enumerate(channels):
                ax = axes[t_idx, c_idx] if n_timesteps > 1 else axes[c_idx]
                
                img = sample_images[t_idx, c_idx]
                
                # Afficher l'image
                if np.isnan(img).all():
                    ax.text(0.5, 0.5, 'NO DATA', ha='center', va='center', 
                           fontsize=12, color='red')
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                else:
                    im = ax.imshow(img, cmap='viridis', aspect='auto')
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                
                # Titre
                if t_idx == 0:
                    ax.set_title(f'{channel}', fontweight='bold')
                if c_idx == 0:
                    ax.set_ylabel(f't{timestep}h', fontweight='bold')
                
                ax.set_xticks([])
                ax.set_yticks([])
        
        plt.tight_layout()
        
        # Sauvegarder
        output_path = filepath.parent / f"sample_{sample_idx}_visualization.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Visualisation sauvegardée: {output_path}")
        
        plt.close()


def check_data_quality(filepath: Path):
    """Vérifie la qualité des données"""
    
    print(f"\n{'='*70}")
    print("VÉRIFICATION DE LA QUALITÉ DES DONNÉES")
    print(f"{'='*70}\n")
    
    with h5py.File(filepath, 'r') as f:
        images = f['images']
        labels = f['labels']
        
        n_samples = len(images)
        channels = f.attrs['channels']
        timesteps = f.attrs['timesteps']
        
        # Vérifier complétude des données
        print("🔍 Complétude des données:")
        
        # Pour chaque canal
        for c_idx, channel in enumerate(channels):
            channel_missing = 0
            for i in range(n_samples):
                sample_imgs = images[i, :, c_idx, :, :]
                if np.isnan(sample_imgs).all():
                    channel_missing += 1
            
            completeness = 100 * (1 - channel_missing / n_samples)
            print(f"  {channel}: {completeness:.1f}% des samples ont au moins 1 image")
        
        # Pour chaque timestep
        print("\n🕐 Disponibilité par timestep:")
        for t_idx, timestep in enumerate(timesteps):
            timestep_complete = 0
            for i in range(n_samples):
                sample_imgs = images[i, t_idx, :, :, :]
                if not np.isnan(sample_imgs).all():
                    timestep_complete += 1
            
            availability = 100 * timestep_complete / n_samples
            print(f"  t{timestep}h: {availability:.1f}% des samples ont des données")
        
        # Samples complètement valides
        print("\n✅ Samples complètement valides:")
        complete_samples = 0
        for i in range(n_samples):
            sample_imgs = images[i]
            sample_lbls = labels[i]
            
            if not np.isnan(sample_imgs).all() and np.isfinite(sample_lbls).any():
                complete_samples += 1
        
        print(f"  {complete_samples}/{n_samples} samples ({100*complete_samples/n_samples:.1f}%) ont images ET labels valides")


def main():
    """Point d'entrée"""
    
    # Chemin du dataset
    if len(sys.argv) > 1:
        dataset_path = Path(sys.argv[1])
    else:
        # Par défaut, chercher le dernier dataset créé
        datasets_dir = Path(r"d:\Documents\Scolarité\5 - INSA Lyon\4TCA\S3\TIP\Projet\weather-prophet\datasets")
        dataset_files = list(datasets_dir.glob("*.h5"))
        
        if not dataset_files:
            print("❌ Aucun dataset trouvé!")
            print(f"Cherché dans: {datasets_dir}")
            return
        
        # Prendre le plus récent
        dataset_path = max(dataset_files, key=lambda p: p.stat().st_mtime)
    
    if not dataset_path.exists():
        print(f"❌ Fichier non trouvé: {dataset_path}")
        return
    
    print(f"📂 Dataset: {dataset_path}")
    print(f"📏 Taille: {dataset_path.stat().st_size / (1024**2):.1f} MB")
    
    # 1. Structure générale
    inspect_hdf5_structure(dataset_path)
    
    # 2. Qualité des données
    check_data_quality(dataset_path)
    
    # 3. Visualiser quelques samples
    print(f"\n{'='*70}")
    print("VISUALISATION DE SAMPLES")
    print(f"{'='*70}")
    
    # Visualiser 3 samples aléatoires
    with h5py.File(dataset_path, 'r') as f:
        n_samples = len(f['images'])
        sample_indices = np.random.choice(n_samples, min(3, n_samples), replace=False)
    
    for idx in sample_indices:
        visualize_sample(dataset_path, idx)
    
    print(f"\n{'='*70}")
    print("✓ INSPECTION TERMINÉE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
