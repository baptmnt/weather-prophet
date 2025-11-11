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
    COMPRESSION = 'gzip'
    COMPRESSION_LEVEL = 4

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
    
    def load_csv(self, zone: str, year: str) -> pd.DataFrame:
        """
        Charge le CSV des stations pour une zone et une année.

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

    def __init__(self, output_path: Path, satellite_dir: Optional[Path] = None, ground_dir: Optional[Path] = None,
                 intermediate_dir: Optional[Path] = None, chunk_size: int = 500, save_intermediate: bool = False):
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

        for i, s in enumerate(self._current_chunk):
            timestamps[i] = str(s['timestamp']).encode('ascii')
            station_ids[i] = int(s['station_id'])
            coords[i, 0] = float(s.get('station_lat', np.nan))
            coords[i, 1] = float(s.get('station_lon', np.nan))
            for ti, tstep in enumerate(Config.TIMESTEPS):
                for ci, ch in enumerate(Config.CHANNELS):
                    im = s['multi_temporal_images'].get(tstep, {}).get(ch)
                    if im is None:
                        images[i, ti, ci, :, :] = np.nan
                    else:
                        images[i, ti, ci, :, :] = im.astype(np.float32)
            for li, var in enumerate(Config.TARGET_VARS):
                v = s['labels'].get(var) if 'labels' in s else s.get(var, np.nan)
                labels[i, li] = np.nan if v is None else float(v)

        out_path = self.intermediate_dir / f"chunk_{self._chunk_idx:05d}.npz"
        np.savez_compressed(str(out_path),
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

        # 3) Construire les samples (itérer ligne par ligne ; chaque ligne = un relevé)
        n_total = 0
        for _, row in stations_df.iterrows():
            ref_time = pd.Timestamp(row['datetime'])
            multi_images = self.satellite_loader.get_multi_temporal_images(ref_time)
            if multi_images is None:
                continue

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

            if self.save_intermediate:
                self._current_chunk.append(sample)
                self._flush_if_needed()
            else:
                self.samples.append(sample)

            n_total += 1

        logger.info(f"Création terminée — samples retenus: {n_total}")

        # 4) Finalisation : écrire intermédiaires ou fichier final
        if self.save_intermediate:
            self._finalize_intermediates()
            merged = self._merge_intermediate_chunks_into_hdf5()
            if not merged:
                logger.error("Erreur lors du merge des fichiers intermédiaires.")
        else:
            # attente : méthode _save_to_hdf5() doit exister dans ce fichier
            if hasattr(self, "_save_to_hdf5"):
                self._save_to_hdf5()
            else:
                logger.error("Méthode _save_to_hdf5() introuvable — impossible d'écrire le fichier final.")

        # 5) Cleanup
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
    args = parser.parse_args()

    # Résoudre dataset_root (par défaut ./data si non fourni)
    dataset_root = Path(args.data_root) if args.data_root else Path('data')
    zone = args.zone

    # Résoudre chemins zone-specific
    sat_dir, ground_dir, out_dir = Config.resolve_zone_paths(dataset_root, zone, args.output_dir)

    # Nom de fichier de sortie (sans date)
    suffix = f"_sta{args.station_id}" if args.station_id else ""
    output_filename = f"meteonet_{zone}_{args.year}{suffix}.h5"
    output_path = out_dir / output_filename

    # intermediate dir resolution
    if args.intermediate_dir:
        inter_dir = Path(args.intermediate_dir)
        if not inter_dir.is_absolute():
            inter_dir = out_dir / inter_dir
    else:
        inter_dir = out_dir / Config.INTERMEDIATE_SUBDIR

    # Construire le dataset en injectant les dossiers d'input spécifiques
    builder = MLDatasetBuilder(output_path, satellite_dir=sat_dir, ground_dir=ground_dir,
                               intermediate_dir=inter_dir, chunk_size=args.chunk_size, save_intermediate=args.save_intermediate)
    
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
    main()
