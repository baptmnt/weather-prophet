"""
Outil d'inspection et validation de dataset HDF5 MeteoNet.

Charge un fichier HDF5 et affiche des informations détaillées sur sa structure,
son contenu et vérifie sa conformité.
"""

import h5py
import numpy as np
from pathlib import Path
import argparse
import logging
from typing import Optional

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DatasetInspector:
    """Inspecte et valide un dataset HDF5 MeteoNet"""
    
    def __init__(self, h5_path: Path):
        self.h5_path = Path(h5_path)
        if not self.h5_path.exists():
            raise FileNotFoundError(f"Dataset non trouvé: {self.h5_path}")
        
        logger.info(f"Ouverture du dataset: {self.h5_path}")
        self.file = h5py.File(self.h5_path, 'r')
    
    def __del__(self):
        if hasattr(self, 'file'):
            self.file.close()
    
    def print_file_structure(self):
        """Affiche la structure complète du fichier HDF5"""
        logger.info("\n" + "="*70)
        logger.info("STRUCTURE DU FICHIER HDF5")
        logger.info("="*70)
        
        def print_item(name, obj, indent=0):
            prefix = "  " * indent
            if isinstance(obj, h5py.Dataset):
                logger.info(f"{prefix}📊 Dataset: {name}")
                logger.info(f"{prefix}   Shape: {obj.shape}, Dtype: {obj.dtype}")
                if obj.compression:
                    logger.info(f"{prefix}   Compression: {obj.compression}")
            elif isinstance(obj, h5py.Group):
                logger.info(f"{prefix}📁 Groupe: {name}")
        
        self.file.visititems(lambda name, obj: print_item(name, obj, name.count('/')))
    
    def print_attributes(self):
        """Affiche les attributs du fichier"""
        logger.info("\n" + "="*70)
        logger.info("ATTRIBUTS DU FICHIER")
        logger.info("="*70)
        
        if len(self.file.attrs) == 0:
            logger.warning("⚠ Aucun attribut trouvé!")
            return
        
        for key, value in self.file.attrs.items():
            if isinstance(value, (list, np.ndarray)):
                logger.info(f"  {key}: {list(value)}")
            else:
                logger.info(f"  {key}: {value}")
    
    def validate_structure(self):
        """Vérifie que la structure attendue est présente"""
        logger.info("\n" + "="*70)
        logger.info("VALIDATION DE LA STRUCTURE")
        logger.info("="*70)
        
        errors = []
        warnings = []
        
        # Vérifier datasets obligatoires
        required_datasets = ['images', 'labels']
        for ds_name in required_datasets:
            if ds_name not in self.file:
                errors.append(f"Dataset manquant: {ds_name}")
            else:
                logger.info(f"  ✓ Dataset '{ds_name}' présent")
        
        # Vérifier groupe metadata
        if 'metadata' not in self.file:
            warnings.append("Groupe 'metadata' manquant")
        else:
            logger.info(f"  ✓ Groupe 'metadata' présent")
            
            # Vérifier contenus metadata
            expected_meta = ['timestamps', 'station_ids', 'station_coords', 'station_heights', 'zones']
            for meta_name in expected_meta:
                full_path = f'metadata/{meta_name}'
                if full_path not in self.file:
                    warnings.append(f"Metadata manquant: {meta_name}")
                else:
                    logger.info(f"    ✓ '{meta_name}' présent")
        
        # Vérifier attributs importants
        expected_attrs = ['n_samples', 'n_timesteps', 'n_channels', 'channels', 'target_vars']
        for attr_name in expected_attrs:
            if attr_name not in self.file.attrs:
                warnings.append(f"Attribut manquant: {attr_name}")
            else:
                logger.info(f"  ✓ Attribut '{attr_name}' présent")
        
        # Afficher résultats
        logger.info("")
        if errors:
            logger.error(f"❌ {len(errors)} erreur(s) critique(s):")
            for err in errors:
                logger.error(f"  - {err}")
        
        if warnings:
            logger.warning(f"⚠ {len(warnings)} avertissement(s):")
            for warn in warnings:
                logger.warning(f"  - {warn}")
        
        if not errors and not warnings:
            logger.info("✅ Structure conforme!")
        
        return len(errors) == 0
    
    def print_dataset_info(self):
        """Affiche les informations détaillées sur le contenu"""
        logger.info("\n" + "="*70)
        logger.info("INFORMATIONS SUR LE DATASET")
        logger.info("="*70)
        
        # Informations générales
        if 'n_samples' in self.file.attrs:
            n_samples = self.file.attrs['n_samples']
            logger.info(f"\n📊 Nombre total de samples: {n_samples}")
        
        # Images
        if 'images' in self.file:
            images = self.file['images']
            logger.info(f"\n🖼️  IMAGES:")
            logger.info(f"  Shape: {images.shape}")
            logger.info(f"  Dtype: {images.dtype}")
            logger.info(f"  Taille: {images.nbytes / (1024**3):.2f} GB")
            
            if 'n_timesteps' in self.file.attrs:
                logger.info(f"  Timesteps: {list(self.file.attrs['timesteps'])}")
            if 'channels' in self.file.attrs:
                logger.info(f"  Canaux: {list(self.file.attrs['channels'])}")
            if 'image_height' in self.file.attrs and 'image_width' in self.file.attrs:
                logger.info(f"  Dimensions: {self.file.attrs['image_height']}×{self.file.attrs['image_width']}")
        
        # Labels
        if 'labels' in self.file:
            labels = self.file['labels']
            logger.info(f"\n🎯 LABELS:")
            logger.info(f"  Shape: {labels.shape}")
            logger.info(f"  Dtype: {labels.dtype}")
            
            if 'target_vars' in self.file.attrs:
                logger.info(f"  Variables: {list(self.file.attrs['target_vars'])}")
        
        # Metadata
        if 'metadata' in self.file:
            logger.info(f"\n📋 METADATA:")
            meta_group = self.file['metadata']
            for key in meta_group.keys():
                ds = meta_group[key]
                logger.info(f"  {key}: shape={ds.shape}, dtype={ds.dtype}")
    
    def compute_statistics(self, sample_size: int = 100):
        """Calcule des statistiques sur un échantillon du dataset"""
        logger.info("\n" + "="*70)
        logger.info(f"STATISTIQUES (échantillon de {sample_size} samples)")
        logger.info("="*70)
        
        if 'images' not in self.file or 'labels' not in self.file:
            logger.error("Impossible de calculer les statistiques: datasets manquants")
            return
        
        n_samples = self.file['images'].shape[0]
        sample_size = min(sample_size, n_samples)
        
        # Échantillonner aléatoirement
        indices = np.random.choice(n_samples, sample_size, replace=False)
        indices = np.sort(indices)
        
        # Images
        logger.info("\n🖼️  STATISTIQUES DES IMAGES:")
        images_sample = self.file['images'][indices]
        
        logger.info(f"  Min global: {np.nanmin(images_sample):.2f}")
        logger.info(f"  Max global: {np.nanmax(images_sample):.2f}")
        logger.info(f"  Mean global: {np.nanmean(images_sample):.2f}")
        logger.info(f"  Std global: {np.nanstd(images_sample):.2f}")
        
        # Pourcentage de NaN
        nan_count = np.isnan(images_sample).sum()
        total_values = images_sample.size
        nan_percent = 100 * nan_count / total_values
        logger.info(f"  NaN: {nan_count}/{total_values} ({nan_percent:.2f}%)")
        
        # Stats par canal si possible
        if 'channels' in self.file.attrs:
            channels = list(self.file.attrs['channels'])
            logger.info(f"\n  Par canal:")
            for c_idx, channel in enumerate(channels):
                channel_data = images_sample[:, :, c_idx, :, :]
                valid_data = channel_data[np.isfinite(channel_data)]
                if len(valid_data) > 0:
                    logger.info(f"    {channel}: mean={valid_data.mean():.2f}, std={valid_data.std():.2f}, "
                              f"min={valid_data.min():.2f}, max={valid_data.max():.2f}")
                else:
                    logger.warning(f"    {channel}: aucune donnée valide")
        
        # Labels
        logger.info("\n🎯 STATISTIQUES DES LABELS:")
        labels_sample = self.file['labels'][indices]
        
        if 'target_vars' in self.file.attrs:
            target_vars = list(self.file.attrs['target_vars'])
            for l_idx, var in enumerate(target_vars):
                var_data = labels_sample[:, l_idx]
                valid_data = var_data[np.isfinite(var_data)]
                
                if len(valid_data) > 0:
                    logger.info(f"  {var}:")
                    logger.info(f"    Valides: {len(valid_data)}/{len(var_data)} ({100*len(valid_data)/len(var_data):.1f}%)")
                    logger.info(f"    Mean: {valid_data.mean():.2f}, Std: {valid_data.std():.2f}")
                    logger.info(f"    Min: {valid_data.min():.2f}, Max: {valid_data.max():.2f}")
                else:
                    logger.warning(f"  {var}: aucune donnée valide")
    
    def check_data_quality(self):
        """Vérifie la qualité des données"""
        logger.info("\n" + "="*70)
        logger.info("CONTRÔLE QUALITÉ")
        logger.info("="*70)
        
        issues = []
        
        # Vérifier cohérence des shapes
        if 'images' in self.file and 'labels' in self.file:
            n_img = self.file['images'].shape[0]
            n_lbl = self.file['labels'].shape[0]
            
            if n_img == n_lbl:
                logger.info(f"  ✓ Cohérence images/labels: {n_img} samples")
            else:
                issues.append(f"Incohérence images ({n_img}) vs labels ({n_lbl})")
        
        # Vérifier metadata
        if 'metadata' in self.file:
            meta_group = self.file['metadata']
            n_samples = self.file['images'].shape[0] if 'images' in self.file else 0
            
            for key in meta_group.keys():
                meta_len = meta_group[key].shape[0]
                if meta_len != n_samples:
                    issues.append(f"Metadata '{key}': {meta_len} entrées vs {n_samples} samples")
                else:
                    logger.info(f"  ✓ Metadata '{key}': {meta_len} entrées")
        
        # Afficher résultats
        if issues:
            logger.error(f"\n❌ {len(issues)} problème(s) détecté(s):")
            for issue in issues:
                logger.error(f"  - {issue}")
            return False
        else:
            logger.info("\n✅ Qualité des données OK!")
            return True
    
    def full_inspection(self, sample_size: int = 100):
        """Lance une inspection complète du dataset"""
        logger.info("\n" + "="*70)
        logger.info(f"INSPECTION COMPLÈTE: {self.h5_path.name}")
        logger.info("="*70)
        logger.info(f"Taille du fichier: {self.h5_path.stat().st_size / (1024**2):.2f} MB")
        
        self.print_file_structure()
        self.print_attributes()
        self.print_dataset_info()
        
        is_valid = self.validate_structure()
        is_quality = self.check_data_quality()
        
        self.compute_statistics(sample_size)
        
        logger.info("\n" + "="*70)
        logger.info("RÉSUMÉ")
        logger.info("="*70)
        if is_valid and is_quality:
            logger.info("✅ Dataset conforme et de bonne qualité")
        elif is_valid:
            logger.warning("⚠️ Dataset conforme mais avec des problèmes de qualité")
        else:
            logger.error("❌ Dataset non conforme")
        logger.info("="*70)


def main():
    """Point d'entrée principal avec arguments CLI"""
    parser = argparse.ArgumentParser(
        description="Inspection et validation de dataset HDF5 MeteoNet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'usage:
  # Inspecter un dataset
  python pytorch_dataloader.py --dataset path/to/meteonet_SE_2016.h5

  # Inspecter avec plus de samples pour les stats
  python pytorch_dataloader.py --dataset path/to/meteonet.h5 --sample-size 500

  # Mode verbeux
  python pytorch_dataloader.py --dataset path/to/meteonet.h5 -v
        """
    )
    
    parser.add_argument('--dataset', type=str, required=False, metavar='PATH',
                        help="Chemin vers le fichier HDF5 (cherche dans ./datasets/ si non fourni)")
    parser.add_argument('--sample-size', type=int, default=100, metavar='N',
                        help="Nombre de samples pour calcul des statistiques (défaut: 100)")
    parser.add_argument('--verbose', '-v', action='store_true',
                        help="Affichage verbeux (niveau DEBUG)")
    parser.add_argument('--quiet', '-q', action='store_true',
                        help="Affichage minimal (niveau WARNING)")
    
    args = parser.parse_args()
    
    # Ajuster logging
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.WARNING)
    
    # Déterminer le chemin du dataset
    if args.dataset:
        dataset_path = Path(args.dataset)
    else:
        # Chercher dans le dossier datasets/
        datasets_dir = Path(__file__).parent.parent / "datasets"
        if datasets_dir.exists():
            h5_files = list(datasets_dir.glob("*.h5"))
            if h5_files:
                dataset_path = h5_files[0]
                logger.info(f"Dataset auto-détecté: {dataset_path}")
            else:
                logger.error("Aucun fichier .h5 trouvé dans ./datasets/")
                return
        else:
            logger.error("Dossier ./datasets/ introuvable. Spécifiez --dataset")
            return
    
    # Lancer l'inspection
    try:
        inspector = DatasetInspector(dataset_path)
        inspector.full_inspection(sample_size=args.sample_size)
    except FileNotFoundError as e:
        logger.error(str(e))
    except Exception as e:
        logger.error(f"Erreur lors de l'inspection: {e}", exc_info=True)


if __name__ == "__main__":
    main()
