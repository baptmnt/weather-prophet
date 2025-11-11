"""
Script complet pour visualiser les fichiers NetCDF satellite
"""
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Configuration
DATA_DIR = Path("D:/Documents/Scolarité/5 - INSA Lyon/4TCA/S3/TIP/Projet/meteonet/data_samples/satellite")

def explore_nc_file(filepath):
    """Explore et affiche les informations d'un fichier NetCDF"""
    print(f"\n{'='*70}")
    print(f"FICHIER: {filepath.name}")
    print(f"{'='*70}")
    
    # Charger avec h5netcdf pour compatibilité Unicode
    data = xr.open_dataset(filepath, engine='h5netcdf')
    
    # Informations générales
    print(f"\nDimensions: {dict(data.dims)}")
    print(f"Variables: {list(data.data_vars)}")
    print(f"Coordonnées: {list(data.coords)}")
    
    # Période temporelle
    if 'time' in data.coords:
        print(f"\nPériode: {data.time.values[0]} → {data.time.values[-1]}")
        print(f"Nombre de pas de temps: {len(data.time)}")
    
    # Zone géographique
    if 'lat' in data.coords and 'lon' in data.coords:
        print(f"\nZone géographique:")
        print(f"  Latitude: {float(data.lat.min()):.2f}° → {float(data.lat.max()):.2f}°")
        print(f"  Longitude: {float(data.lon.min()):.2f}° → {float(data.lon.max()):.2f}°")
    
    # Statistiques sur la première variable
    var_name = list(data.data_vars)[0]
    var = data[var_name]
    print(f"\nStatistiques sur '{var_name}':")
    print(f"  Min: {float(var.min()):.2f}")
    print(f"  Max: {float(var.max()):.2f}")
    print(f"  Moyenne: {float(var.mean()):.2f}")
    
    return data

def visualize_timeseries(data, var_name, num_frames=4):
    """Visualise plusieurs pas de temps d'une variable"""
    var = data[var_name]
    
    # Sélectionner des indices régulièrement espacés
    n_times = len(data.time)
    indices = np.linspace(0, n_times-1, num_frames, dtype=int)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        ax = axes[i]
        time_val = data.time.values[idx]
        
        im = var.isel(time=idx).plot(ax=ax, add_colorbar=False)
        ax.set_title(f"{var_name} - {time_val}")
    
    plt.colorbar(im, ax=axes, orientation='horizontal', pad=0.05, label=var_name)
    plt.tight_layout()
    plt.savefig(f'{var_name}_timeseries.png', dpi=150, bbox_inches='tight')
    print(f"\nGraphique sauvegardé: {var_name}_timeseries.png")
    plt.close()

def compare_files():
    """Compare tous les fichiers satellites disponibles"""
    nc_files = list(DATA_DIR.glob("*.nc"))
    
    print(f"\n{'='*70}")
    print(f"FICHIERS DISPONIBLES ({len(nc_files)})")
    print(f"{'='*70}")
    
    for nc_file in nc_files:
        print(f"\n• {nc_file.name}")
        data = xr.open_dataset(nc_file, engine='h5netcdf')
        var_name = list(data.data_vars)[0]
        print(f"  Variable: {var_name}")
        print(f"  Dimensions: {data[var_name].shape}")
        print(f"  Période: {len(data.time)} pas de temps")

if __name__ == "__main__":
    # 1. Comparer tous les fichiers
    compare_files()
    
    # 2. Explorer en détail le fichier CT
    ct_file = DATA_DIR / "CT_NW_2016.nc"
    data = explore_nc_file(ct_file)
    
    # 3. Visualiser la série temporelle
    visualize_timeseries(data, 'CT', num_frames=4)
    
    print("\n" + "="*70)
    print("TERMINÉ!")
    print("="*70)
