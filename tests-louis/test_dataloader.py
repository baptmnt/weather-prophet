"""
Test rapide du DataLoader - version simplifiée
"""

from pytorch_dataloader import MeteoNetDataset, create_dataloaders
from pathlib import Path
import torch

print("="*70)
print("TEST RAPIDE DU DATALOADER")
print("="*70)

# Chemin du dataset
dataset_path = Path(r"d:\Documents\Scolarité\5 - INSA Lyon\4TCA\S3\TIP\Projet\weather-prophet\datasets\meteonet_SE_2016_20160101.h5")

print(f"\n📂 Dataset: {dataset_path.name}")
print(f"📏 Taille: {dataset_path.stat().st_size / (1024**2):.1f} MB")

# Créer les DataLoaders
print("\n🔄 Création des DataLoaders...")
train_loader, val_loader, test_loader = create_dataloaders(
    dataset_path,
    batch_size=16,
    train_split=0.7,
    val_split=0.15,
    num_workers=0  # 0 pour éviter les problèmes Windows
)

print("\n✅ DataLoaders créés!")
print(f"  Train: {len(train_loader)} batches")
print(f"  Val: {len(val_loader)} batches")
print(f"  Test: {len(test_loader)} batches")

# Tester un batch
print("\n🧪 Test de chargement d'un batch...")
images, labels, metadata = next(iter(train_loader))

print(f"\n📊 Résultats:")
print(f"  Images shape: {images.shape}")
print(f"  Labels shape: {labels.shape}")
print(f"  Images - min: {images.min():.2f}, max: {images.max():.2f}, mean: {images.mean():.2f}")

# Filtrer les NaN pour les labels
labels_valid = labels[~torch.isnan(labels)]
if len(labels_valid) > 0:
    print(f"  Labels - min: {labels_valid.min():.2f}, max: {labels_valid.max():.2f}, mean: {labels_valid.mean():.2f}")
    print(f"  Labels - NaN count: {torch.isnan(labels).sum().item()}/{labels.numel()} ({100*torch.isnan(labels).sum().item()/labels.numel():.1f}%)")

print(f"\n📍 Metadata du premier sample:")
print(f"  Timestamp: {metadata['timestamp'][0]}")
print(f"  Station ID: {metadata['station_id'][0]}")
print(f"  Zone: {metadata['zone'][0]}")

print("\n" + "="*70)
print("✅ TOUT FONCTIONNE PARFAITEMENT!")
print("="*70)
print("\n🚀 Vous pouvez maintenant:")
print("  1. Créer un modèle PyTorch")
print("  2. Itérer sur train_loader pour l'entraînement")
print("  3. Évaluer sur val_loader et test_loader")
print("\nBon ML! 🧠💪")
