"""
Génération du dataset ML au format HDF5 pour la prédiction météo.

Structure du dataset:
- Inputs: Images satellites complètes (CT, IR039, IR108, VIS06, WV062) à t-12h, t-24h, t-48h, t-168h
- Outputs: Mesures stations au sol (t, hu, precip, dd, ff, psl, td)

Usage:
    python create_ml_dataset.py
"""

import xarray as xr
import pandas as pd
import numpy as np
import h5py
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Config:
    """Configuration du dataset"""
    # Chemins
    SATELLITE_DIR = Path(r"d:\Documents\Scolarité\5 - INSA Lyon\4TCA\S3\TIP\Projet\meteonet\data_samples\satellite")
    GROUND_STATIONS_DIR = Path(r"d:\Documents\Scolarité\5 - INSA Lyon\4TCA\S3\TIP\Projet\meteonet\data_samples\ground_stations")
    OUTPUT_DIR = Path(r"d:\Documents\Scolarité\5 - INSA Lyon\4TCA\S3\TIP\Projet\weather-prophet\datasets")
    
    # Paramètres temporels (en heures)
    TIMESTEPS = [-12, -24, -48, -168]  # t-12h, t-1j, t-2j, t-7j
    
    # Canaux satellites
    CHANNELS = ['CT', 'IR039', 'IR108', 'VIS06', 'WV062']
    
    # Variables météo de sortie (dans l'ordre du CSV)
    TARGET_VARS = ['dd', 'ff', 'precip', 'hu', 'td', 't', 'psl']
    
    # Format de sortie
    OUTPUT_FORMAT = 'hdf5'
    COMPRESSION = 'gzip'
    COMPRESSION_LEVEL = 4  # 0-9, compromis vitesse/taille
    
    # Zones disponibles
    ZONES = ['NW', 'SE']


class SatelliteDataLoader:
    """Charge et gère les données satellites NetCDF"""
    
    def __init__(self, satellite_dir: Path):
        self.satellite_dir = satellite_dir
        self.datasets: Dict[str, xr.Dataset] = {}
        self.image_cache: Dict[str, np.ndarray] = {}
        
    def load_satellite_files(self, zone: str, year: int) -> Dict[str, xr.Dataset]:
        """Charge tous les fichiers satellites d'une zone/année"""
        logger.info(f"Chargement des fichiers satellites pour {zone}_{year}")
        
        datasets = {}
        for channel in Config.CHANNELS:
            filename = f"{channel}_{zone}_{year}.nc"
            filepath = self.satellite_dir / filename
            
            if filepath.exists():
                try:
                    ds = xr.open_dataset(filepath, engine='h5netcdf')
                    datasets[channel] = ds
                    logger.info(f"  ✓ {channel}: {len(ds.time)} timesteps, shape {ds[channel].shape}")
                except Exception as e:
                    logger.warning(f"  ✗ {channel}: Erreur de chargement - {e}")
            else:
                logger.warning(f"  ✗ {channel}: Fichier non trouvé")
        
        self.datasets = datasets
        return datasets
    
    def get_image_at_time(self, channel: str, target_time: pd.Timestamp) -> Optional[np.ndarray]:
        """Récupère l'image d'un canal à un timestamp donné"""
        if channel not in self.datasets:
            return None
        
        ds = self.datasets[channel]
        
        # Créer une clé de cache
        cache_key = f"{channel}_{target_time}"
        if cache_key in self.image_cache:
            return self.image_cache[cache_key]
        
        try:
            # Trouver le timestamp le plus proche
            time_diffs = np.abs(ds.time.values - target_time.to_datetime64())
            closest_idx = np.argmin(time_diffs)
            closest_time = pd.Timestamp(ds.time.values[closest_idx])
            
            # Vérifier si la différence est acceptable (< 1h)
            time_diff_minutes = abs((target_time - closest_time).total_seconds() / 60)
            if time_diff_minutes > 60:
                logger.debug(f"Pas de données {channel} proche de {target_time} (écart: {time_diff_minutes:.0f} min)")
                return None
            
            # Extraire l'image
            image = ds[channel].isel(time=closest_idx).values
            
            # Gérer les valeurs manquantes
            if np.isnan(image).all():
                return None
            
            # Cache l'image
            self.image_cache[cache_key] = image
            
            return image
            
        except Exception as e:
            logger.debug(f"Erreur lors de l'extraction de {channel} à {target_time}: {e}")
            return None
    
    def get_multi_temporal_images(self, reference_time: pd.Timestamp) -> Optional[Dict[str, np.ndarray]]:
        """
        Récupère les images de tous les canaux pour les 4 timesteps passés.
        
        Returns:
            Dict avec structure: {timestep: {channel: image_array}}
            Exemple: {-12: {'CT': array, 'IR039': array, ...}, -24: {...}, ...}
        """
        multi_temporal = {}
        
        for timestep_offset in Config.TIMESTEPS:
            target_time = reference_time + pd.Timedelta(hours=timestep_offset)
            
            timestep_images = {}
            for channel in Config.CHANNELS:
                image = self.get_image_at_time(channel, target_time)
                if image is not None:
                    timestep_images[channel] = image
            
            # On garde le timestep seulement si au moins 1 canal est disponible
            if timestep_images:
                multi_temporal[timestep_offset] = timestep_images
        
        # Vérifier qu'on a au moins quelques timesteps
        if len(multi_temporal) < 2:
            return None
        
        return multi_temporal
    
    def close(self):
        """Ferme tous les datasets"""
        for ds in self.datasets.values():
            ds.close()
        self.datasets.clear()
        self.image_cache.clear()


class GroundStationDataLoader:
    """Charge et gère les données des stations au sol"""
    
    def __init__(self, ground_stations_dir: Path):
        self.ground_stations_dir = ground_stations_dir
        self.data: Optional[pd.DataFrame] = None
    
    def load_csv(self, zone: str, date: str) -> pd.DataFrame:
        """
        Charge le CSV des stations pour une zone et une date.
        
        Args:
            zone: 'NW' ou 'SE'
            date: Format 'YYYYMMDD' (ex: '20160101')
        """
        filename = f"{zone}_{date}.csv"
        filepath = self.ground_stations_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Fichier stations non trouvé: {filepath}")
        
        logger.info(f"Chargement des stations: {filename}")
        
        # Charger le CSV
        df = pd.read_csv(filepath)
        
        # Convertir la date en datetime
        df['datetime'] = pd.to_datetime(df['date'], format='%Y%m%d %H:%M')
        
        # Trier par station et temps
        df = df.sort_values(['number_sta', 'datetime'])
        
        logger.info(f"  {len(df)} mesures, {df['number_sta'].nunique()} stations uniques")
        
        self.data = df
        return df
    
    def get_measurement_at_time(self, station_id: int, target_time: pd.Timestamp) -> Optional[Dict]:
        """Récupère les mesures d'une station à un timestamp donné"""
        if self.data is None:
            return None
        
        # Filtrer par station
        station_data = self.data[self.data['number_sta'] == station_id]
        if station_data.empty:
            return None
        
        # Trouver le timestamp le plus proche
        time_diffs = np.abs(station_data['datetime'] - target_time)
        closest_idx = time_diffs.idxmin()
        
        # Vérifier si l'écart est acceptable (< 30 min pour les stations)
        time_diff_minutes = time_diffs[closest_idx].total_seconds() / 60
        if time_diff_minutes > 30:
            return None
        
        row = station_data.loc[closest_idx]
        
        # Extraire les variables cibles
        measurement = {
            'station_id': int(row['number_sta']),
            'lat': float(row['lat']),
            'lon': float(row['lon']),
            'height_sta': float(row['height_sta']),
            'datetime': row['datetime'],
        }
        
        # Ajouter les variables météo (gérer les NaN)
        for var in Config.TARGET_VARS:
            value = row[var]
            measurement[var] = float(value) if pd.notna(value) else np.nan
        
        return measurement


class MLDatasetBuilder:
    """Construit le dataset HDF5 pour le ML"""
    
    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.satellite_loader: Optional[SatelliteDataLoader] = None
        self.station_loader: Optional[GroundStationDataLoader] = None
        
        self.samples: List[Dict] = []
        self.unique_images: Dict[str, np.ndarray] = {}  # Pour optimiser le stockage
        
    def build_dataset(self, zone: str, year: int, date: str):
        """
        Construit le dataset pour une zone/année/date.
        
        Args:
            zone: 'NW' ou 'SE'
            year: Année (ex: 2016)
            date: Date au format 'YYYYMMDD' (ex: '20160101')
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"Construction du dataset pour {zone}_{year} - {date}")
        logger.info(f"{'='*70}\n")
        
        # 1. Charger les données satellites
        self.satellite_loader = SatelliteDataLoader(Config.SATELLITE_DIR)
        satellite_datasets = self.satellite_loader.load_satellite_files(zone, year)
        
        if not satellite_datasets:
            logger.error("Aucune donnée satellite chargée!")
            return
        
        # 2. Charger les données stations
        self.station_loader = GroundStationDataLoader(Config.GROUND_STATIONS_DIR)
        try:
            stations_df = self.station_loader.load_csv(zone, date)
        except FileNotFoundError as e:
            logger.error(str(e))
            return
        
        # 3. Créer les samples
        logger.info("\nCréation des samples...")
        self._create_samples(stations_df, zone)
        
        # 4. Sauvegarder en HDF5
        if self.samples:
            logger.info(f"\nSauvegarde de {len(self.samples)} samples dans {self.output_path}")
            self._save_to_hdf5()
            logger.info("✓ Dataset créé avec succès!")
        else:
            logger.warning("Aucun sample valide créé!")
        
        # 5. Nettoyage
        self.satellite_loader.close()
    
    def _create_samples(self, stations_df: pd.DataFrame, zone: str):
        """Crée les samples à partir des données stations et satellites"""
        
        # Obtenir les timestamps uniques
        unique_times = stations_df['datetime'].unique()
        logger.info(f"Timestamps à traiter: {len(unique_times)}")
        
        # Obtenir les stations uniques
        unique_stations = stations_df['number_sta'].unique()
        logger.info(f"Stations à traiter: {len(unique_stations)}")
        
        total_attempts = 0
        successful_samples = 0
        
        # Pour chaque timestamp
        for time_idx, target_time in enumerate(unique_times):
            target_time_pd = pd.Timestamp(target_time)
            
            # Récupérer les images multi-temporelles
            multi_temporal_images = self.satellite_loader.get_multi_temporal_images(target_time_pd)
            
            if multi_temporal_images is None:
                logger.debug(f"Pas assez d'images satellites pour {target_time_pd}")
                continue
            
            # Pour chaque station à ce timestamp
            for station_id in unique_stations:
                total_attempts += 1
                
                # Récupérer les mesures de la station
                measurement = self.station_loader.get_measurement_at_time(station_id, target_time_pd)
                
                if measurement is None:
                    continue
                
                # Vérifier qu'on a au moins quelques variables non-NaN
                labels = [measurement[var] for var in Config.TARGET_VARS]
                if sum(np.isfinite(labels)) < 3:  # Au moins 3 variables valides
                    continue
                
                # Créer le sample
                sample = {
                    'timestamp': target_time_pd,
                    'station_id': station_id,
                    'station_lat': measurement['lat'],
                    'station_lon': measurement['lon'],
                    'station_height': measurement['height_sta'],
                    'zone': zone,
                    'multi_temporal_images': multi_temporal_images,
                    'labels': labels,
                }
                
                self.samples.append(sample)
                successful_samples += 1
            
            # Log de progression
            if (time_idx + 1) % 10 == 0:
                logger.info(f"  Traité {time_idx + 1}/{len(unique_times)} timestamps - {successful_samples} samples valides")
        
        logger.info(f"\nRésumé création samples:")
        logger.info(f"  Tentatives: {total_attempts}")
        logger.info(f"  Succès: {successful_samples}")
        logger.info(f"  Taux de réussite: {100*successful_samples/total_attempts:.1f}%")
    
    def _save_to_hdf5(self):
        """Sauvegarde les samples dans un fichier HDF5 optimisé"""
        
        with h5py.File(self.output_path, 'w') as f:
            n_samples = len(self.samples)
            
            # Déterminer les dimensions des images (depuis le premier sample)
            first_sample = self.samples[0]
            first_timestep = list(first_sample['multi_temporal_images'].keys())[0]
            first_channel = list(first_sample['multi_temporal_images'][first_timestep].keys())[0]
            sample_image = first_sample['multi_temporal_images'][first_timestep][first_channel]
            img_h, img_w = sample_image.shape
            
            n_timesteps = len(Config.TIMESTEPS)
            n_channels = len(Config.CHANNELS)
            n_labels = len(Config.TARGET_VARS)
            
            logger.info(f"Dimensions du dataset:")
            logger.info(f"  Samples: {n_samples}")
            logger.info(f"  Timesteps: {n_timesteps}")
            logger.info(f"  Channels: {n_channels}")
            logger.info(f"  Image shape: ({img_h}, {img_w})")
            logger.info(f"  Labels: {n_labels}")
            
            # Créer les datasets HDF5
            # Images: (n_samples, n_timesteps, n_channels, height, width)
            images_dset = f.create_dataset(
                'images',
                shape=(n_samples, n_timesteps, n_channels, img_h, img_w),
                dtype='float32',
                compression=Config.COMPRESSION,
                compression_opts=Config.COMPRESSION_LEVEL,
                chunks=(1, n_timesteps, n_channels, img_h, img_w)  # 1 sample à la fois
            )
            
            # Labels: (n_samples, n_labels)
            labels_dset = f.create_dataset(
                'labels',
                shape=(n_samples, n_labels),
                dtype='float32',
                compression=Config.COMPRESSION,
            )
            
            # Metadata
            metadata_grp = f.create_group('metadata')
            
            # Timestamps (stockés comme strings)
            timestamps = [str(s['timestamp']) for s in self.samples]
            metadata_grp.create_dataset('timestamps', data=np.array(timestamps, dtype='S26'))
            
            # Station IDs
            station_ids = np.array([s['station_id'] for s in self.samples], dtype='int32')
            metadata_grp.create_dataset('station_ids', data=station_ids)
            
            # Coordonnées stations
            coords = np.array([[s['station_lat'], s['station_lon']] for s in self.samples], dtype='float32')
            metadata_grp.create_dataset('station_coords', data=coords)
            
            # Hauteurs stations
            heights = np.array([s['station_height'] for s in self.samples], dtype='float32')
            metadata_grp.create_dataset('station_heights', data=heights)
            
            # Zones
            zones = [s['zone'] for s in self.samples]
            metadata_grp.create_dataset('zones', data=np.array(zones, dtype='S2'))
            
            # Attributs (configuration)
            f.attrs['n_samples'] = n_samples
            f.attrs['n_timesteps'] = n_timesteps
            f.attrs['n_channels'] = n_channels
            f.attrs['n_labels'] = n_labels
            f.attrs['image_height'] = img_h
            f.attrs['image_width'] = img_w
            f.attrs['timesteps'] = Config.TIMESTEPS
            f.attrs['channels'] = Config.CHANNELS
            f.attrs['target_vars'] = Config.TARGET_VARS
            f.attrs['creation_date'] = str(datetime.now())
            
            # Remplir les données
            logger.info("\nÉcriture des données...")
            for idx, sample in enumerate(self.samples):
                # Préparer les images pour ce sample
                sample_images = np.full((n_timesteps, n_channels, img_h, img_w), np.nan, dtype='float32')
                
                for t_idx, timestep in enumerate(Config.TIMESTEPS):
                    if timestep in sample['multi_temporal_images']:
                        for c_idx, channel in enumerate(Config.CHANNELS):
                            if channel in sample['multi_temporal_images'][timestep]:
                                img = sample['multi_temporal_images'][timestep][channel]
                                sample_images[t_idx, c_idx] = img
                
                images_dset[idx] = sample_images
                labels_dset[idx] = sample['labels']
                
                if (idx + 1) % 100 == 0:
                    logger.info(f"  Écrit {idx + 1}/{n_samples} samples")
            
            logger.info("✓ Écriture terminée")


def main():
    """Point d'entrée principal"""
    
    # Configuration
    zone = 'SE'  # Peut être changé en 'NW'
    year = 2016
    date = '20160101'  # Date du CSV des stations
    
    output_filename = f"meteonet_{zone}_{year}_{date}.h5"
    output_path = Config.OUTPUT_DIR / output_filename
    
    # Construire le dataset
    builder = MLDatasetBuilder(output_path)
    builder.build_dataset(zone, year, date)
    
    logger.info("\n" + "="*70)
    logger.info("TERMINÉ!")
    logger.info(f"Dataset sauvegardé: {output_path}")
    logger.info(f"Taille: {output_path.stat().st_size / (1024**2):.1f} MB")
    logger.info("="*70)


if __name__ == "__main__":
    main()
