import xarray as xr
from pathlib import Path

# Charger le fichier NetCDF
# IMPORTANT: Utiliser engine='h5netcdf' pour compatibilité avec Python 3.13 et chemins Unicode
path = Path("D:/Documents/Scolarité/5 - INSA Lyon/4TCA/S3/TIP/Projet/meteonet/data_samples/satellite/CT_NW_2016.nc")
data = xr.open_dataset(path, engine='h5netcdf')

# Afficher les informations du fichier
print("="*70)
print("INFORMATIONS DU FICHIER NETCDF")
print("="*70)
print(data)
print("\n" + "="*70)
print("RESUME")
print("="*70)
print(f"Variables: {list(data.data_vars)}")
print(f"Coordonnees: {list(data.coords)}")
print(f"Nombre de pas de temps: {len(data.time)}")
print(f"Dimensions spatiales: {len(data.lat)} x {len(data.lon)}")
print(f"Periode: de {data.time.values[0]} a {data.time.values[-1]}")
