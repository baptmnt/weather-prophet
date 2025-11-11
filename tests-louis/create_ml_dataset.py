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
import argparse
import os
import glob
import bisect
from multiprocessing import Pool, cpu_count, freeze_support
from functools import partial
try:
    import dask  # facultatif
    HAS_DASK = True
except Exception:
    HAS_DASK = False
    # Pas bloquant: on restera en mode xarray classique (pas de message ici pour éviter le bruit lorsqu'on n'utilise pas Dask)

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Config:
    """Configuration du dataset (chemins configurables)."""
    # Racine par défaut (attendu : data/<ZONE>/...)
    DATASET_ROOT = Path('data')

    # Sous-dossiers standardisés à l'intérieur du dossier de zone (zone/satellite, zone/ground_stations, ...)
    SATELLITE_SUBDIR = Path("satellite")
    GROUND_STATIONS_SUBDIR = Path("ground_stations")
    OUTPUT_SUBDIR = Path("datasets")
    INTERMEDIATE_SUBDIR = Path("intermediate")  # new: sous-dossier pour fichiers intermédiaires

    # Paramètres temporels (en heures)
    TIMESTEPS = [-12, -24, -48, -168]  # t-12h, t-1j, t-2j, t-7j

    # Canaux satellites
    CHANNELS = ['CT', 'IR039', 'IR108', 'VIS06', 'WV062']

    # Variables météo de sortie (dans l'ordre du CSV)
    TARGET_VARS = ['dd', 'ff', 'precip', 'hu', 'td', 't', 'psl']

    # Format de sortie / compression
    OUTPUT_FORMAT = 'hdf5'
    COMPRESSION = 'gzip'  # Nécessaire pour fichier raisonnable (16 GB → 92 MB pour 4767 samples)
    COMPRESSION_LEVEL = 4

    # Lecture optimisée NetCDF
    USE_DASK = False           # Activer xarray+dask (lazy loading) pour les NetCDF
    DASK_TIME_CHUNK = 256      # Taille de chunk sur l'axe temps (ex: 256)
    PRELOAD_IMAGES = False     # Précharger en mémoire toutes les images nécessaires (réduit I/O)

    # Downsampling spatial
    DOWNSAMPLE_FACTOR = 1      # Facteur de réduction spatiale (1=original, 2=moitié, 5=1/5, 10=1/10)

    # Zones disponibles
    ZONES = ['NW', 'SE']

    @classmethod
    def resolve_zone_paths(cls, dataset_root: Path, zone: str, output_dir: Optional[str] = None):
        """
        Retourne (satellite_dir, ground_stations_dir, output_dir_path) pour la zone donnée.
        - dataset_root : dossier qui contient les dossiers de zone (ex: data/)
        - zone : 'SE' ou 'NW'
        - output_dir : optionnel ; si None -> dataset_root/zone/OUTPUT_SUBDIR
        """
        base = Path(dataset_root).expanduser().resolve()
        zone_dir = base / zone
        sat_dir = zone_dir / cls.SATELLITE_SUBDIR
        ground_dir = zone_dir / cls.GROUND_STATIONS_SUBDIR

        if output_dir:
            # Si c'est absolu, on l'utilise tel quel
            # Sinon, on part du répertoire courant qui est différent du dataset root
            if os.path.isabs(output_dir):
                out_dir = Path(output_dir)
            else:
                out_dir = Path.cwd() / output_dir
                out_dir = out_dir.resolve()
            
        else:
            out_dir = zone_dir / cls.OUTPUT_SUBDIR

        return sat_dir, ground_dir, out_dir


class SatelliteDataLoader:
    """Charge et gère les données satellites NetCDF avec pré-indexation temporelle"""
    
    def __init__(self, satellite_dir: Path):
        self.satellite_dir = satellite_dir
        self.datasets: Dict[str, xr.Dataset] = {}
        self.image_cache: Dict[str, np.ndarray] = {}
        # Pré-indexation: channel -> {timestamp: index}
        self.time_indices: Dict[str, Dict[pd.Timestamp, int]] = {}
        # Liste triée des timestamps pour recherche rapide (bisect)
        self.sorted_times: Dict[str, List[pd.Timestamp]] = {}
        
    def load_satellite_files(self, zone: str, year: int) -> Dict[str, xr.Dataset]:
        """Charge tous les fichiers satellites d'une zone/année et construit l'index temporel"""
        logger.info(f"Chargement des fichiers satellites pour {zone}_{year}")
        
        datasets = {}
        for channel in Config.CHANNELS:
            filename = f"{channel}_{zone}_{year}.nc"
            filepath = self.satellite_dir / filename
            
            if filepath.exists():
                try:
                    open_kwargs = {"engine": "h5netcdf"}
                    # Activer dask si demandé et disponible
                    if Config.USE_DASK and HAS_DASK:
                        open_kwargs["chunks"] = {"time": Config.DASK_TIME_CHUNK}
                        logger.info(f"  → {channel}: ouverture avec Dask chunks time={Config.DASK_TIME_CHUNK}")
                    elif Config.USE_DASK and not HAS_DASK:
                        logger.warning("Dask non installé: ouverture sans chunks (désactiver --use-dask ou installer dask)")
                    ds = xr.open_dataset(filepath, **open_kwargs)
                    datasets[channel] = ds
                    
                    # Pré-indexation: créer un dictionnaire timestamp -> index
                    time_index = {}
                    timestamps = []
                    for idx, time_val in enumerate(ds.time.values):
                        timestamp = pd.Timestamp(time_val)
                        time_index[timestamp] = idx
                        timestamps.append(timestamp)
                    
                    self.time_indices[channel] = time_index
                    self.sorted_times[channel] = sorted(timestamps)
                    
                    logger.info(f"  ✓ {channel}: {len(ds.time)} timesteps indexés, shape {ds[channel].shape}")
                except Exception as e:
                    logger.warning(f"  ✗ {channel}: Erreur de chargement - {e}")
            else:
                logger.warning(f"  ✗ {channel}: Fichier non trouvé")
        
        self.datasets = datasets
        return datasets
    
    def get_image_at_time(self, channel: str, target_time: pd.Timestamp) -> Optional[np.ndarray]:
        """Récupère l'image d'un canal à un timestamp donné (avec recherche dichotomique O(log n))"""
        if channel not in self.datasets:
            return None
        
        # Créer une clé de cache
        cache_key = f"{channel}_{target_time}"
        if cache_key in self.image_cache:
            return self.image_cache[cache_key]
        
        try:
            sorted_times = self.sorted_times.get(channel, [])
            time_index = self.time_indices.get(channel, {})
            
            if not sorted_times:
                # Fallback: pas d'index disponible
                logger.debug(f"Pas d'index temporel pour {channel}")
                return None
            
            # Recherche dichotomique du timestamp le plus proche (O(log n))
            pos = bisect.bisect_left(sorted_times, target_time)
            
            # Trouver le timestamp le plus proche parmi les voisins
            candidates = []
            if pos > 0:
                candidates.append((sorted_times[pos - 1], abs((target_time - sorted_times[pos - 1]).total_seconds())))
            if pos < len(sorted_times):
                candidates.append((sorted_times[pos], abs((target_time - sorted_times[pos]).total_seconds())))
            
            if not candidates:
                return None
            
            # Sélectionner le plus proche
            closest_time, time_diff_seconds = min(candidates, key=lambda x: x[1])
            time_diff_minutes = time_diff_seconds / 60
            
            # Vérifier si la différence est acceptable (< 1h)
            if time_diff_minutes > 60:
                logger.debug(f"Pas de données {channel} proche de {target_time} (écart: {time_diff_minutes:.0f} min)")
                return None
            
            # Récupérer l'index et extraire l'image
            closest_idx = time_index[closest_time]
            ds = self.datasets[channel]
            image = ds[channel].isel(time=closest_idx).values
            
            # Gérer les valeurs manquantes
            if np.isnan(image).all():
                return None
            
            # Appliquer le downsampling spatial si demandé
            if Config.DOWNSAMPLE_FACTOR > 1:
                # Average pooling pour préserver les valeurs moyennes (pas besoin de skimage)
                factor = Config.DOWNSAMPLE_FACTOR
                h, w = image.shape
                new_h, new_w = h // factor, w // factor
                # Reshape et moyenne sur les blocs
                image = image[:new_h*factor, :new_w*factor].reshape(new_h, factor, new_w, factor).mean(axis=(1, 3))
            
            # Cache l'image
            self.image_cache[cache_key] = image
            
            return image
            
        except Exception as e:
            logger.debug(f"Erreur lors de l'extraction de {channel} à {target_time}: {e}")
            return None
    
    def preload_images_for_timestamps(self, timestamps: List[pd.Timestamp], channel: str) -> Dict[pd.Timestamp, np.ndarray]:
        """
        Précharge plusieurs images d'un canal pour des timestamps donnés (I/O optimisé).
        
        Args:
            timestamps: Liste des timestamps à précharger
            channel: Canal satellite
            
        Returns:
            Dict {timestamp: image_array} des images chargées
        """
        if channel not in self.datasets:
            return {}
        
        ds = self.datasets[channel]
        time_index = self.time_indices.get(channel, {})
        sorted_times = self.sorted_times.get(channel, [])
        
        if not sorted_times:
            return {}
        
        # Trouver les indices correspondants pour tous les timestamps
        indices_to_load = []
        timestamp_mapping = {}  # {index: original_timestamp}
        
        for target_time in timestamps:
            cache_key = f"{channel}_{target_time}"
            if cache_key in self.image_cache:
                continue  # Déjà en cache
            
            # Recherche dichotomique
            pos = bisect.bisect_left(sorted_times, target_time)
            
            candidates = []
            if pos > 0:
                candidates.append((sorted_times[pos - 1], abs((target_time - sorted_times[pos - 1]).total_seconds())))
            if pos < len(sorted_times):
                candidates.append((sorted_times[pos], abs((target_time - sorted_times[pos]).total_seconds())))
            
            if candidates:
                closest_time, time_diff_seconds = min(candidates, key=lambda x: x[1])
                if time_diff_seconds / 60 <= 60:  # < 1h
                    idx = time_index[closest_time]
                    indices_to_load.append(idx)
                    timestamp_mapping[idx] = target_time
        
        # Charger toutes les images en un seul appel (vectorisé)
        if indices_to_load:
            try:
                # Avec Dask activé, on force le chargement en mémoire pour ces indices
                var = ds[channel].isel(time=indices_to_load)
                images = var.load().values if (Config.USE_DASK and HAS_DASK) else var.values
                
                # Mettre en cache
                for i, idx in enumerate(indices_to_load):
                    target_time = timestamp_mapping[idx]
                    image = images[i]
                    
                    if not np.isnan(image).all():
                        cache_key = f"{channel}_{target_time}"
                        self.image_cache[cache_key] = image
            except Exception as e:
                logger.debug(f"Erreur lors du préchargement batch de {channel}: {e}")
        
        return {}
    
    def get_multi_temporal_images(self, reference_time: pd.Timestamp) -> Optional[Dict[str, np.ndarray]]:
        """
        Récupère les images de tous les canaux pour les 4 timesteps passés.
        Optimisé avec cache pour réutilisation sur plusieurs stations.
        
        Returns:
            Dict avec structure: {timestep: {channel: image_array}}
            Exemple: {-12: {'CT': array, 'IR039': array, ...}, -24: {...}, ...}
        """
        # Clé de cache pour tout le set d'images multi-temporelles
        cache_key = f"multi_{reference_time}"
        if cache_key in self.image_cache:
            return self.image_cache[cache_key]
        
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
        
        # Cache le résultat complet pour réutilisation
        self.image_cache[cache_key] = multi_temporal
        
        return multi_temporal
    
    def close(self):
        """Ferme tous les datasets"""
        for ds in self.datasets.values():
            ds.close()
        self.datasets.clear()
        self.image_cache.clear()
        self.time_indices.clear()
        self.sorted_times.clear()


class GroundStationDataLoader:
    """Charge et gère les données des stations au sol avec pré-indexation"""
    
    def __init__(self, ground_stations_dir: Path):
        self.ground_stations_dir = ground_stations_dir
        self.data: Optional[pd.DataFrame] = None
        # Pré-indexation: (station_id, timestamp) -> row_index
        self.station_time_index: Dict[Tuple[int, pd.Timestamp], int] = {}
        # Index par station pour recherche rapide
        self.station_times: Dict[int, List[pd.Timestamp]] = {}
    
    def load_csv(self, zone: str, year: str) -> pd.DataFrame:
        """
        Charge le CSV des stations pour une zone et une année avec indexation.

        Args:
            zone: 'NW' ou 'SE'
            year: Année (ex: 2016)
        """
        filename = f"{zone}{year}_single.csv" #TODO EDIT
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
        
        # Construire l'index (station_id, timestamp) -> row_index
        logger.info(f"Construction de l'index temporel des stations...")
        for idx, row in df.iterrows():
            station_id = int(row['number_sta'])
            timestamp = pd.Timestamp(row['datetime'])
            self.station_time_index[(station_id, timestamp)] = idx
            
            # Ajouter à la liste des timestamps par station
            if station_id not in self.station_times:
                self.station_times[station_id] = []
            self.station_times[station_id].append(timestamp)
        
        # Trier les timestamps par station pour recherche dichotomique
        for station_id in self.station_times:
            self.station_times[station_id] = sorted(self.station_times[station_id])
        
        logger.info(f"  {len(df)} mesures, {df['number_sta'].nunique()} stations indexées")
        
        self.data = df
        return df
    
    def get_measurement_at_time(self, station_id: int, target_time: pd.Timestamp) -> Optional[Dict]:
        """Récupère les mesures d'une station à un timestamp donné (avec recherche dichotomique)"""
        if self.data is None:
            return None
        
        # Vérifier si la station existe
        if station_id not in self.station_times:
            return None
        
        sorted_times = self.station_times[station_id]
        
        # Recherche dichotomique du timestamp le plus proche (O(log n))
        pos = bisect.bisect_left(sorted_times, target_time)
        
        # Trouver le timestamp le plus proche parmi les voisins
        candidates = []
        if pos > 0:
            candidates.append((sorted_times[pos - 1], abs((target_time - sorted_times[pos - 1]).total_seconds())))
        if pos < len(sorted_times):
            candidates.append((sorted_times[pos], abs((target_time - sorted_times[pos]).total_seconds())))
        
        if not candidates:
            return None
        
        # Sélectionner le plus proche
        closest_time, time_diff_seconds = min(candidates, key=lambda x: x[1])
        time_diff_minutes = time_diff_seconds / 60
        
        # Vérifier si l'écart est acceptable (< 30 min pour les stations)
        if time_diff_minutes > 30:
            return None
        
        # Récupérer la ligne via l'index
        row_idx = self.station_time_index[(station_id, closest_time)]
        row = self.data.loc[row_idx]
        
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
    
    def close(self):
        """Ferme les ressources (ici rien à fermer pour un DataFrame)"""
        pass


# ============================================================================
# Fonctions standalone pour parallélisation (picklable)
# ============================================================================

def process_timestamp_batch(batch_data: Tuple[List[Tuple], Path, str, int]) -> List[Dict]:
    """
    Traite un batch de timestamps en parallèle.
    Fonction standalone pour être utilisable avec multiprocessing.
    
    Args:
        batch_data: Tuple contenant (timestamp_groups, satellite_dir, zone, year)
            - timestamp_groups: Liste de (ref_time, group_dict) où group_dict est un dict de listes
            - satellite_dir: Chemin vers les fichiers satellites
            - zone: Zone géographique
            - year: Année
    
    Returns:
        Liste de samples générés
    """
    timestamp_groups, satellite_dir, zone, year = batch_data
    
    # Créer un loader satellite pour ce worker
    sat_loader = SatelliteDataLoader(satellite_dir)
    sat_loader.load_satellite_files(zone, year)
    
    samples = []
    
    for ref_time, group_dict in timestamp_groups:
        ref_time = pd.Timestamp(ref_time)
        
        # Charger les images satellites UNE SEULE FOIS pour ce timestamp
        multi_images = sat_loader.get_multi_temporal_images(ref_time)
        if multi_images is None:
            continue
        
        # Traiter TOUTES les stations de ce timestamp avec les MÊMES images
        # Convertir group_dict en DataFrame pour itération
        n_rows = len(group_dict['number_sta'])
        for i in range(n_rows):
            # Extraire les valeurs de cette ligne
            row = {key: group_dict[key][i] for key in group_dict.keys()}
            
            # construire labels dict
            labels = {}
            valid_label_count = 0
            for var in Config.TARGET_VARS:
                v = row.get(var, np.nan)
                labels[var] = np.nan if pd.isna(v) else float(v)
                if not pd.isna(v):
                    valid_label_count += 1

            # appliquer critère minimal (au moins 1 label valide ici, adapter si besoin)
            if valid_label_count < 1:
                continue

            sample = {
                'timestamp': ref_time,
                'station_id': int(row['number_sta']),
                'station_lat': float(row.get('lat', np.nan)),
                'station_lon': float(row.get('lon', np.nan)),
                'multi_temporal_images': multi_images,
                'labels': labels
            }
            
            samples.append(sample)
    
    # Nettoyer
    sat_loader.close()
    
    return samples


class MLDatasetBuilder:
    """Construit le dataset HDF5 pour le ML"""

    def __init__(self, output_path: Path, satellite_dir: Optional[Path] = None, ground_dir: Optional[Path] = None,
                 intermediate_dir: Optional[Path] = None, chunk_size: int = 500, save_intermediate: bool = False,
                 num_workers: int = 1):
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Permet d'injecter des chemins d'input différents pour la zone
        self.satellite_dir = satellite_dir or Config.DATASET_ROOT / "satellite"
        self.ground_dir = ground_dir or Config.DATASET_ROOT / "ground_stations"

        # intermediate handling
        self.save_intermediate = save_intermediate
        if intermediate_dir is None:
            self.intermediate_dir = self.output_path.parent / Config.INTERMEDIATE_SUBDIR
        else:
            self.intermediate_dir = Path(intermediate_dir)
        if self.save_intermediate:
            self.intermediate_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = int(chunk_size)
        self._current_chunk = []
        self._chunk_idx = 0
        
        # Parallélisation
        self.num_workers = max(1, min(num_workers, cpu_count()))  # Entre 1 et nombre de CPUs

        self.satellite_loader: Optional[SatelliteDataLoader] = None
        self.station_loader: Optional[GroundStationDataLoader] = None

        self.samples: List[Dict] = []
        self.unique_images: Dict[str, np.ndarray] = {}  # Pour optimiser le stockage

    # flush current buffer of samples to an intermediate NPZ file
    def _write_intermediate_chunk(self):
        if not self._current_chunk:
            return
        # build arrays from samples
        first = self._current_chunk[0]
        first_timestep = list(first['multi_temporal_images'].keys())[0]
        first_channel = list(first['multi_temporal_images'][first_timestep].keys())[0]
        sample_image = first['multi_temporal_images'][first_timestep][first_channel]
        img_h, img_w = sample_image.shape
        n_timesteps = len(Config.TIMESTEPS)
        n_channels = len(Config.CHANNELS)
        n_labels = len(Config.TARGET_VARS)

        k = len(self._current_chunk)
        images = np.zeros((k, n_timesteps, n_channels, img_h, img_w), dtype=np.float32)
        labels = np.full((k, n_labels), np.nan, dtype=np.float32)
        timestamps = np.empty(k, dtype='S26')
        station_ids = np.empty(k, dtype=np.int32)
        coords = np.zeros((k, 2), dtype=np.float32)

        # OPTIMISATION : Vectorisation de la conversion dict → array
        # Au lieu de copier pixel par pixel, on stack les images directement
        for i, s in enumerate(self._current_chunk):
            timestamps[i] = str(s['timestamp']).encode('ascii')
            station_ids[i] = int(s['station_id'])
            coords[i, 0] = float(s.get('station_lat', np.nan))
            coords[i, 1] = float(s.get('station_lon', np.nan))
            
            # Conversion optimisée : construire une liste d'images puis stack
            for ti, tstep in enumerate(Config.TIMESTEPS):
                for ci, ch in enumerate(Config.CHANNELS):
                    im = s['multi_temporal_images'].get(tstep, {}).get(ch)
                    if im is not None:
                        # Copie directe du tableau entier (beaucoup plus rapide)
                        images[i, ti, ci] = im.astype(np.float32)
                    # Sinon reste à 0 (ou np.nan si préféré)
            
            # Labels
            for li, var in enumerate(Config.TARGET_VARS):
                v = s['labels'].get(var) if 'labels' in s else s.get(var, np.nan)
                labels[i, li] = np.nan if v is None else float(v)

        out_path = self.intermediate_dir / f"chunk_{self._chunk_idx:05d}.npz"
        # OPTIMISATION : np.savez au lieu de savez_compressed (beaucoup plus rapide)
        # La compression sera faite lors du merge final dans HDF5
        np.savez(str(out_path),
                 images=images, labels=labels,
                 timestamps=timestamps, station_ids=station_ids, coords=coords)
        logger.info(f"Wrote intermediate chunk {out_path} ({k} samples)")
        self._chunk_idx += 1
        self._current_chunk = []

    def _flush_if_needed(self):
        if not self.save_intermediate:
            return
        if len(self._current_chunk) >= self.chunk_size:
            self._write_intermediate_chunk()

    def _finalize_intermediates(self):
        # write any remaining
        if self.save_intermediate and self._current_chunk:
            self._write_intermediate_chunk()

    def _merge_intermediate_chunks_into_hdf5(self, start_idx: Optional[int] = None, end_idx: Optional[int] = None) -> bool:
        """Lit les .npz intermédiaires et écrit le fichier final HDF5.
        Permet de ne merger qu'un sous-ensemble de chunks via start_idx / end_idx (indices inclusifs).
        """
        files = sorted(glob.glob(str(self.intermediate_dir / "chunk_*.npz")))
        if not files:
            logger.warning("Aucun fichier intermédiaire trouvé pour merger.")
            return False

        # Normaliser indices
        n_files = len(files)
        s = 0 if start_idx is None else max(0, int(start_idx))
        e = n_files - 1 if end_idx is None else min(n_files - 1, int(end_idx))
        if s > e:
            logger.error(f"Indices de merge invalides: start={s} > end={e}")
            return False

        sel_files = files[s:e+1]
        logger.info(f"Merging intermediate chunks [{s}:{e}] -> {len(sel_files)} files")

        # compute total samples
        total = 0
        for fpath in sel_files:
            logger.info(f"  Lecture chunk: {Path(fpath).name}")
            with np.load(fpath) as npz:
                k = npz['images'].shape[0]
                total += k
                logger.info(f"    → {k} samples")

        # get shapes from first selected file
        logger.info(f"\nTotal à merger: {total} samples")
        with np.load(sel_files[0]) as npz:
            k0, n_timesteps, n_channels, h, w = npz['images'].shape
            n_labels = npz['labels'].shape[1]
            logger.info(f"Shape images: ({n_timesteps}, {n_channels}, {h}, {w}), labels: {n_labels}")

        # create HDF5 (ensure parent exists)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"\nCréation du fichier HDF5: {self.output_path}")
        
        with h5py.File(self.output_path, 'w') as f:
            images_dset = f.create_dataset('images',
                                          shape=(total, n_timesteps, n_channels, h, w),
                                          dtype='float32',
                                          compression=Config.COMPRESSION,
                                          compression_opts=Config.COMPRESSION_LEVEL)
            labels_dset = f.create_dataset('labels', shape=(total, n_labels), dtype='float32',
                                          compression=Config.COMPRESSION)
            timestamps = f.create_dataset('timestamps', shape=(total,), dtype='S26')
            station_ids = f.create_dataset('station_ids', shape=(total,), dtype='int32')
            coords = f.create_dataset('coords', shape=(total, 2), dtype='float32')

            offset = 0
            for i, fpath in enumerate(sel_files):
                logger.info(f"  Merge chunk {i+1}/{len(sel_files)}: {Path(fpath).name}")
                with np.load(fpath) as npz:
                    k = npz['images'].shape[0]
                    images_dset[offset:offset + k] = npz['images']
                    labels_dset[offset:offset + k] = npz['labels']
                    timestamps[offset:offset + k] = npz['timestamps']
                    station_ids[offset:offset + k] = npz['station_ids']
                    coords[offset:offset + k] = npz['coords']
                    logger.info(f"    → écrit à offset {offset}, {k} samples")
                    offset += k
                    
        logger.info(f"\n✓ Merged {len(sel_files)} chunks → {self.output_path} ({total} samples)")
        return True
    
    def _save_to_hdf5(self):
        """Sauvegarde les samples en mémoire directement dans le fichier HDF5 final"""
        if not self.samples:
            logger.warning("Aucun sample à sauvegarder")
            return
        
        n_samples = len(self.samples)
        logger.info(f"\nSauvegarde de {n_samples} samples dans {self.output_path}")
        
        # Récupérer les shapes depuis le premier sample
        first = self.samples[0]
        first_timestep = list(first['multi_temporal_images'].keys())[0]
        first_channel = list(first['multi_temporal_images'][first_timestep].keys())[0]
        sample_image = first['multi_temporal_images'][first_timestep][first_channel]
        img_h, img_w = sample_image.shape
        n_timesteps = len(Config.TIMESTEPS)
        n_channels = len(Config.CHANNELS)
        n_labels = len(Config.TARGET_VARS)
        
        # Construire les arrays en mémoire d'abord (plus rapide)
        logger.info(f"  Construction des arrays en mémoire...")
        images = np.zeros((n_samples, n_timesteps, n_channels, img_h, img_w), dtype=np.float32)
        labels = np.full((n_samples, n_labels), np.nan, dtype=np.float32)
        timestamps_arr = np.empty(n_samples, dtype='S26')
        station_ids_arr = np.empty(n_samples, dtype=np.int32)
        coords_arr = np.zeros((n_samples, 2), dtype=np.float32)
        
        # Remplir les arrays
        for i, sample in enumerate(self.samples):
            # Convertir multi_temporal_images (dict) en array - OPTIMISÉ
            for ti, tstep in enumerate(Config.TIMESTEPS):
                for ci, ch in enumerate(Config.CHANNELS):
                    im = sample['multi_temporal_images'].get(tstep, {}).get(ch)
                    if im is not None:
                        # Copie directe du tableau entier (rapide)
                        images[i, ti, ci] = im.astype(np.float32)
                    # Sinon reste à 0
            
            # Labels
            for li, var in enumerate(Config.TARGET_VARS):
                v = sample['labels'].get(var) if 'labels' in sample else sample.get(var, np.nan)
                labels[i, li] = np.nan if v is None else float(v)
            
            # Metadata
            timestamps_arr[i] = str(sample['timestamp']).encode('utf-8')
            station_ids_arr[i] = sample['station_id']
            coords_arr[i] = [sample['station_lat'], sample['station_lon']]
            
            if (i + 1) % 1000 == 0:
                logger.info(f"    Progression: {i + 1}/{n_samples} samples traités")
        
        # Créer le fichier HDF5 et écrire d'un coup
        logger.info(f"  Écriture dans {self.output_path}...")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(self.output_path, 'w') as f:
            f.create_dataset('images', data=images,
                           compression=Config.COMPRESSION,
                           compression_opts=Config.COMPRESSION_LEVEL)
            f.create_dataset('labels', data=labels,
                           compression=Config.COMPRESSION)
            f.create_dataset('timestamps', data=timestamps_arr)
            f.create_dataset('station_ids', data=station_ids_arr)
            f.create_dataset('coords', data=coords_arr)
        
        logger.info(f"✓ {n_samples} samples sauvegardés dans {self.output_path}")

    def build_dataset(self, zone: str, year: int, station_id: Optional[int] = None):
        """
        Construit le dataset pour une zone/année.
        Si station_id est fourni, ne conserve que les relevés de cette station.
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"Construction du dataset pour {zone}_{year} (station_filter={station_id})")
        logger.info(f"{'='*70}\n")

        # 1) Charger les données satellites
        self.satellite_loader = SatelliteDataLoader(self.satellite_dir)
        sat_datasets = self.satellite_loader.load_satellite_files(zone, year)
        if not sat_datasets:
            logger.error("Aucune donnée satellite chargée — arrêt.")
            return

        # 2) Charger les données stations
        self.station_loader = GroundStationDataLoader(self.ground_dir)
        try:
            stations_df = self.station_loader.load_csv(zone, year)
        except FileNotFoundError as e:
            logger.error(str(e))
            return

        # 2.b Filtrer sur la station cible si demandé
        if station_id is not None:
            stations_df = stations_df[stations_df['number_sta'] == int(station_id)]
            if stations_df.empty:
                logger.warning(f"Aucune donnée trouvée pour la station {station_id}. Arrêt.")
                return
            logger.info(f"Filtrage : {len(stations_df)} lignes pour la station {station_id}")

        # 3) Construire les samples avec VECTORISATION par timestamp
        # Grouper par timestamp unique pour éviter de recharger les mêmes images
        logger.info(f"Vectorisation : groupement par timestamps uniques...")
        grouped = stations_df.groupby('datetime')
        unique_timestamps = len(grouped)
        logger.info(f"  {unique_timestamps} timestamps uniques à traiter")

        # Optionnel: précharger en mémoire toutes les images nécessaires (toutes les combinaisons timestamp+timesteps)
        if Config.PRELOAD_IMAGES:
            logger.info("Préchargement des images nécessaires (tous canaux, tous timesteps)...")
            unique_ref_times = [pd.Timestamp(k) for k in grouped.groups.keys()]
            all_target_times = set()
            for ref_time in unique_ref_times:
                for dt in Config.TIMESTEPS:
                    all_target_times.add(ref_time + pd.Timedelta(hours=dt))
            all_target_times = sorted(all_target_times)
            logger.info(f"  → {len(all_target_times)} timestamps à précharger par canal")
            for ch in Config.CHANNELS:
                self.satellite_loader.preload_images_for_timestamps(all_target_times, ch)
        
        # Déterminer les dimensions depuis les datasets chargés
        img_h, img_w = None, None
        for ch in Config.CHANNELS:
            if ch in sat_datasets:
                ds_shape = sat_datasets[ch][ch].shape
                if len(ds_shape) == 3:
                    _, img_h, img_w = ds_shape
                    # Appliquer le downsampling aux dimensions
                    if Config.DOWNSAMPLE_FACTOR > 1:
                        img_h = img_h // Config.DOWNSAMPLE_FACTOR
                        img_w = img_w // Config.DOWNSAMPLE_FACTOR
                    break
        
        if img_h is None or img_w is None:
            logger.error("Impossible de déterminer les dimensions des images")
            return
        
        logger.info(f"  Dimensions images finales: {img_h}×{img_w} pixels")
        
        n_timesteps = len(Config.TIMESTEPS)
        n_channels = len(Config.CHANNELS)
        n_labels = len(Config.TARGET_VARS)
        
        # PHASE 1 : Compter le nombre RÉEL de samples valides (rapide, pas d'allocation)
        logger.info(f"  Phase 1/3 : Comptage des samples valides...")
        n_valid = 0
        for ref_time, group_df in grouped:
            multi_images = self.satellite_loader.get_multi_temporal_images(ref_time)
            if not multi_images:
                continue
            for _, row in group_df.iterrows():
                sta_id = int(row['number_sta'])
                measurement = self.station_loader.get_measurement_at_time(sta_id, ref_time)
                if measurement:
                    n_valid += 1
        
        logger.info(f"  → {n_valid} samples valides détectés")
        
        # PHASE 2 : Pré-allouer les arrays EXACTS (pas de gaspillage mémoire)
        logger.info(f"  Phase 2/3 : Pré-allocation des arrays ({n_valid} samples)...")
        images_array = np.zeros((n_valid, n_timesteps, n_channels, img_h, img_w), dtype=np.float32)
        labels_array = np.full((n_valid, n_labels), np.nan, dtype=np.float32)
        timestamps_array = np.empty(n_valid, dtype='S26')
        station_ids_array = np.empty(n_valid, dtype=np.int32)
        coords_array = np.zeros((n_valid, 2), dtype=np.float32)
        
        # PHASE 3 : Remplir directement les arrays (pas de copies, pas de listes)
        logger.info(f"  Phase 3/3 : Remplissage des arrays...")
        n_filled = 0
        progress_step = max(1, unique_timestamps // 10)
        
        for idx, (ref_time, group_df) in enumerate(grouped):
            # Récupérer les images satellites pour ce timestamp (une seule fois)
            multi_images = self.satellite_loader.get_multi_temporal_images(ref_time)
            if not multi_images:
                continue
            
            # Convertir multi_images (dict) en array une seule fois pour ce timestamp
            ts_images = np.zeros((n_timesteps, n_channels, img_h, img_w), dtype=np.float32)
            for ti, tstep in enumerate(Config.TIMESTEPS):
                for ci, ch in enumerate(Config.CHANNELS):
                    im = multi_images.get(tstep, {}).get(ch)
                    if im is not None:
                        ts_images[ti, ci] = im.astype(np.float32)
            
            # Traiter toutes les stations de ce timestamp
            for _, row in group_df.iterrows():
                sta_id = int(row['number_sta'])
                
                # Récupérer les mesures au sol
                measurement = self.station_loader.get_measurement_at_time(sta_id, ref_time)
                if not measurement:
                    continue
                
                # Écrire DIRECTEMENT dans les arrays pré-alloués (pas de .copy() !)
                images_array[n_filled] = ts_images  # Assign direct, numpy gère la copie
                
                for li, var in enumerate(Config.TARGET_VARS):
                    val = measurement.get(var, np.nan)
                    labels_array[n_filled, li] = np.nan if val is None else float(val)
                
                timestamps_array[n_filled] = str(ref_time).encode('ascii')
                station_ids_array[n_filled] = sta_id
                coords_array[n_filled] = [measurement.get('lat', np.nan), measurement.get('lon', np.nan)]
                
                n_filled += 1
            
            if (idx + 1) % progress_step == 0:
                logger.info(f"    Progression: {idx + 1}/{unique_timestamps} timestamps ({n_filled} samples)")
        
        logger.info(f"✓ Création terminée — {n_filled} samples créés")
        
        # 4) Écriture directe en HDF5 (une seule fois, rapide)
        logger.info(f"\nÉcriture HDF5 : {n_filled} samples → {self.output_path}")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(self.output_path, 'w') as f:
            # Compression conditionnelle
            if Config.COMPRESSION:
                f.create_dataset('images', data=images_array,
                               compression=Config.COMPRESSION,
                               compression_opts=Config.COMPRESSION_LEVEL)
                f.create_dataset('labels', data=labels_array,
                               compression=Config.COMPRESSION)
            else:
                f.create_dataset('images', data=images_array)
                f.create_dataset('labels', data=labels_array)
            
            f.create_dataset('timestamps', data=timestamps_array)
            f.create_dataset('station_ids', data=station_ids_array)
            f.create_dataset('coords', data=coords_array)
        
        logger.info(f"✓ Dataset écrit : {self.output_path}")

        # Cleanup
        if self.satellite_loader:
            self.satellite_loader.close()
        if self.station_loader:
            self.station_loader.close()


def main():
    """Point d'entrée principal — arguments CLI pour dossiers relatifs/configurables (sans --date)"""
    parser = argparse.ArgumentParser(description="Génération du dataset ML (satellite + stations).")
    parser.add_argument('--data-root', type=str, required=False,
                        help="Dossier racine contenant les dossiers de zones (ex: 'data').")
    parser.add_argument('--zone', type=str, required=False, default='SE', choices=Config.ZONES,
                        help="Zone à traiter (ex: SE)")
    parser.add_argument('--year', type=int, required=False, default=2016, help="Année des fichiers satellites")
    parser.add_argument('--output-dir', type=str, required=False,
                        help="Dossier de sortie relatif à data-root/ZONE ou chemin absolu. Si non fourni -> data-root/ZONE/datasets")
    parser.add_argument('--station-id', type=int, required=False,
                        help="Si fourni, ne conserver que les relevés de cette station (number_sta).")
    parser.add_argument('--save-intermediate', action='store_true',
                        help="Si activé, écrit des fichiers intermédiaires (chunks .npz) pendant la construction.")
    parser.add_argument('--build-final', action='store_true',
                        help="Si activé, écrit des fichiers finaux (dataset.h5) à partir des chunks existants.")
    parser.add_argument('--merge-start', type=int, required=False,
                        help="Index du premier chunk à merger (0-based, inclusif).")
    parser.add_argument('--merge-end', type=int, required=False,
                        help="Index du dernier chunk à merger (0-based, inclusif).")
    parser.add_argument('--intermediate-dir', type=str, required=False,
                        help="Chemin pour stocker les fichiers intermédiaires (si relatif, relatif à output-dir).")
    parser.add_argument('--chunk-size', type=int, default=500,
                        help="Taille des chunks (nombre de samples) pour écriture intermédiaire.")
    parser.add_argument('--num-workers', type=int, default=None,
                        help=f"Nombre de processus parallèles (défaut: 1, max: {cpu_count()}). Utiliser 0 pour auto (tous les CPUs). Si non spécifié, une question sera posée.")
    parser.add_argument('--use-dask', action='store_true',
                        help="Activer Dask pour lecture lazy des NetCDF (chunks sur l'axe temps).")
    parser.add_argument('--dask-chunk-time', type=int, required=False,
                        help="Taille du chunk Dask sur l'axe temps (ex: 256).")
    parser.add_argument('--preload-images', action='store_true',
                        help="Précharger en mémoire toutes les images nécessaires (réduit fortement les I/O).")
    parser.add_argument('--compression-level', type=int, required=False,
                        help="Niveau de compression gzip (0-9). 0 = pas de compression (fichier énorme), 1 = rapide, 9 = maximum.")
    parser.add_argument('--downsample-factor', type=int, required=False, choices=[1, 2, 5, 10], default=1,
                        help="Facteur de réduction spatiale des images (1=original 171×261, 2=moitié 86×131, 5=1/5 34×52, 10=1/10 17×26). Réduit la taille du fichier et accélère l'entraînement.")
    args = parser.parse_args()

    # Résoudre dataset_root (par défaut ./data si non fourni)
    dataset_root = Path(args.data_root) if args.data_root else Path('data')
    zone = args.zone

    # Résoudre chemins zone-specific
    sat_dir, ground_dir, out_dir = Config.resolve_zone_paths(dataset_root, zone, args.output_dir)

    # Nom de fichier de sortie (sans date)
    suffix = f"_sta{args.station_id}" if args.station_id else ""
    # Ajouter suffixe downsampling si actif
    if args.downsample_factor > 1:
        suffix += f"_ds{args.downsample_factor}"
    output_filename = f"meteonet_{zone}_{args.year}{suffix}.h5"
    output_path = out_dir / output_filename

    # intermediate dir resolution
    if args.intermediate_dir:
        inter_dir = Path(args.intermediate_dir)
        if not inter_dir.is_absolute():
            inter_dir = out_dir / inter_dir
    else:
        inter_dir = out_dir / Config.INTERMEDIATE_SUBDIR

    # Résoudre num_workers avec question interactive si non spécifié
    if args.num_workers is None:
        print("\n" + "="*70)
        print("⚙️  CONFIGURATION DE LA PARALLÉLISATION")
        print("="*70)
        print(f"\nVotre machine dispose de {cpu_count()} CPU(s).")
        print("\nOptions de parallélisation:")
        print("  • 1 worker  : Mode séquentiel (recommandé pour Windows, stable)")
        print("  • 2-4 workers : Parallélisation modérée (peut ralentir sur Windows)")
        print("  • 0 (auto)  : Tous les CPUs disponibles")
        print("\n⚠️  Note: Sur Windows, la parallélisation ajoute un overhead significatif")
        print("   et peut être PLUS LENTE que le mode séquentiel. Le mode séquentiel")
        print("   est déjà très rapide grâce aux optimisations (pré-indexation + vectorisation).")
        
        while True:
            try:
                response = input(f"\nNombre de workers à utiliser [défaut: 1] : ").strip()
                if response == "":
                    num_workers = 1
                    break
                num_workers_input = int(response)
                if num_workers_input == 0:
                    num_workers = cpu_count()
                    print(f"✓ Utilisation de tous les CPUs ({cpu_count()} workers)")
                    break
                elif 1 <= num_workers_input <= cpu_count():
                    num_workers = num_workers_input
                    break
                else:
                    print(f"❌ Valeur invalide. Choisissez entre 0 et {cpu_count()}.")
            except ValueError:
                print("❌ Veuillez entrer un nombre entier.")
        
        print(f"\n✓ Configuration: {num_workers} worker{'s' if num_workers > 1 else ''}")
        print("="*70 + "\n")
    else:
        # num_workers spécifié en argument
        num_workers = cpu_count() if args.num_workers == 0 else args.num_workers

    # Appliquer options d'optimisation NetCDF
    if args.use_dask:
        Config.USE_DASK = True
    if args.dask_chunk_time:
        Config.DASK_TIME_CHUNK = int(args.dask_chunk_time)
    if args.preload_images:
        Config.PRELOAD_IMAGES = True
    if args.compression_level is not None:
        lvl = max(0, min(9, int(args.compression_level)))
        Config.COMPRESSION_LEVEL = lvl
    
    # Appliquer downsampling spatial
    Config.DOWNSAMPLE_FACTOR = args.downsample_factor
    if Config.DOWNSAMPLE_FACTOR > 1:
        logger.info(f"Downsampling spatial activé: facteur {Config.DOWNSAMPLE_FACTOR} (réduction {Config.DOWNSAMPLE_FACTOR**2}× en taille)")

    # Construire le dataset en injectant les dossiers d'input spécifiques
    builder = MLDatasetBuilder(output_path, satellite_dir=sat_dir, ground_dir=ground_dir,
                               intermediate_dir=inter_dir, chunk_size=args.chunk_size, 
                               save_intermediate=args.save_intermediate, num_workers=num_workers)
    
    if args.build_final:
        print("Construction du fichier final à partir des chunks intermédiaires existants...")
        builder._merge_intermediate_chunks_into_hdf5(start_idx=args.merge_start, end_idx=args.merge_end)

    else: 
        builder.build_dataset(zone, args.year, station_id=args.station_id)

        # if intermediate saving requested, finalize and merge
        if args.save_intermediate:
            builder._finalize_intermediates()
            builder._merge_intermediate_chunks_into_hdf5()
    
    

    logger.info("\n" + "="*70)
    logger.info("TERMINÉ!")
    logger.info(f"Dataset sauvegardé: {output_path}")
    if output_path.exists():
        logger.info(f"Taille: {output_path.stat().st_size / (1024**2):.1f} MB")
    logger.info("="*70)


if __name__ == "__main__":
    freeze_support()  # Nécessaire pour Windows avec multiprocessing
    main()
