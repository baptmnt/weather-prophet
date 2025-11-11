"""
Test simple pour mesurer les gains de l'optimisation de pré-indexation temporelle.
"""

import time
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
import bisect

# Configuration
SATELLITE_DIR = Path(r"d:\Documents\Scolarité\5 - INSA Lyon\4TCA\S3\TIP\Projet\meteonet\data_samples\satellite")
CHANNEL = "IR108"
ZONE = "SE"
YEAR = 2016

print("="*70)
print("TEST D'OPTIMISATION - PRÉ-INDEXATION TEMPORELLE")
print("="*70)

# Charger le fichier satellite
filepath = SATELLITE_DIR / f"{CHANNEL}_{ZONE}_{YEAR}.nc"
print(f"\n📂 Chargement: {filepath.name}")

if not filepath.exists():
    print(f"❌ Fichier non trouvé: {filepath}")
    exit(1)

ds = xr.open_dataset(filepath, engine='h5netcdf')
print(f"   ✓ {len(ds.time)} timesteps chargés")

# Créer l'index temporel (nouvelle méthode optimisée)
print("\n🔧 Construction de l'index temporel...")
start_index = time.time()

time_index = {}
sorted_times = []
for idx, time_val in enumerate(ds.time.values):
    timestamp = pd.Timestamp(time_val)
    time_index[timestamp] = idx
    sorted_times.append(timestamp)

sorted_times = sorted(sorted_times)
end_index = time.time()

print(f"   ✓ Index créé en {(end_index - start_index)*1000:.2f}ms")
print(f"   ✓ {len(sorted_times)} timestamps indexés")

# Créer des timestamps de test aléatoires
print("\n🎯 Génération de 1000 timestamps de test...")
np.random.seed(42)
first_time = pd.Timestamp(ds.time.values[0])
last_time = pd.Timestamp(ds.time.values[-1])
time_range_seconds = (last_time - first_time).total_seconds()

test_timestamps = []
for _ in range(1000):
    random_offset = np.random.uniform(0, time_range_seconds)
    test_time = first_time + pd.Timedelta(seconds=random_offset)
    test_timestamps.append(test_time)

print(f"   ✓ {len(test_timestamps)} timestamps générés")

# TEST 1: Méthode ANCIENNE (np.argmin sur tous les timestamps)
print("\n" + "="*70)
print("TEST 1: Méthode ANCIENNE (np.argmin - O(n))")
print("="*70)

start_old = time.time()
results_old = []

for target_time in test_timestamps:
    time_diffs = np.abs(ds.time.values - target_time.to_datetime64())
    closest_idx = np.argmin(time_diffs)
    results_old.append(closest_idx)

end_old = time.time()
elapsed_old = end_old - start_old

print(f"⏱️  Temps total: {elapsed_old*1000:.2f}ms")
print(f"⏱️  Temps moyen par recherche: {elapsed_old/len(test_timestamps)*1000:.4f}ms")
print(f"🔍 {len(results_old)} recherches effectuées")

# TEST 2: Méthode NOUVELLE (bisect - O(log n))
print("\n" + "="*70)
print("TEST 2: Méthode NOUVELLE (bisect - O(log n))")
print("="*70)

start_new = time.time()
results_new = []

for target_time in test_timestamps:
    # Recherche dichotomique
    pos = bisect.bisect_left(sorted_times, target_time)
    
    # Trouver le plus proche parmi les voisins
    candidates = []
    if pos > 0:
        candidates.append((sorted_times[pos - 1], abs((target_time - sorted_times[pos - 1]).total_seconds())))
    if pos < len(sorted_times):
        candidates.append((sorted_times[pos], abs((target_time - sorted_times[pos]).total_seconds())))
    
    if candidates:
        closest_time, _ = min(candidates, key=lambda x: x[1])
        closest_idx = time_index[closest_time]
        results_new.append(closest_idx)

end_new = time.time()
elapsed_new = end_new - start_new

print(f"⏱️  Temps total: {elapsed_new*1000:.2f}ms")
print(f"⏱️  Temps moyen par recherche: {elapsed_new/len(test_timestamps)*1000:.4f}ms")
print(f"🔍 {len(results_new)} recherches effectuées")

# COMPARAISON
print("\n" + "="*70)
print("📊 RÉSULTATS")
print("="*70)

speedup = elapsed_old / elapsed_new
time_saved = elapsed_old - elapsed_new

print(f"\n🚀 Gain de performance:")
print(f"   • Méthode ancienne: {elapsed_old*1000:.2f}ms")
print(f"   • Méthode nouvelle: {elapsed_new*1000:.2f}ms")
print(f"   • Accélération: {speedup:.1f}x plus rapide")
print(f"   • Temps économisé: {time_saved*1000:.2f}ms pour 1000 recherches")

# Vérifier que les résultats sont identiques
print(f"\n✅ Validation:")
identical = np.array_equal(results_old, results_new)
print(f"   • Résultats identiques: {'OUI ✓' if identical else 'NON ✗'}")

# Projection pour 83,000 samples avec 4 timesteps
total_searches = 83000 * 4  # 4 timesteps par sample
# Ajoutons aussi les 5 canaux satellites
total_searches_all_channels = total_searches * 5  # 5 canaux

projected_old = (elapsed_old / 1000) * total_searches_all_channels / 1000
projected_new = (elapsed_new / 1000) * total_searches_all_channels / 1000

print(f"\n🎯 PROJECTION POUR 83,000 SAMPLES:")
print(f"   • 4 timesteps × 5 canaux = 20 recherches par sample")
print(f"   • Total de recherches: {total_searches_all_channels:,}")
print(f"   • Méthode ancienne: {projected_old:.2f}s ({projected_old/60:.2f}min)")
print(f"   • Méthode nouvelle: {projected_new:.2f}s ({projected_new/60:.2f}min)")
print(f"   • Temps économisé: {(projected_old - projected_new):.2f}s (~{(projected_old - projected_new)/60:.1f}min)")

# Avec un dataset réel d'un an (environ 8760 heures = 8760*2 timestamps avec images toutes les 30min)
realistic_timestamps = 8760 * 2  # environ 17,520 timestamps pour une année
print(f"\n🌍 PROJECTION RÉALISTE (année complète avec {realistic_timestamps:,} timestamps):")
# Recalculer avec plus de timestamps (impact O(n) vs O(log n) est plus visible)
speedup_factor = realistic_timestamps / len(sorted_times)  # Facteur d'augmentation
projected_old_real = projected_old * (speedup_factor ** 0.5)  # O(n) croît linéairement
projected_new_real = projected_new * np.log2(realistic_timestamps) / np.log2(len(sorted_times))  # O(log n) croît logarithmiquement

print(f"   • Méthode ancienne estimée: {projected_old_real:.2f}s ({projected_old_real/60:.2f}min)")
print(f"   • Méthode nouvelle estimée: {projected_new_real:.2f}s ({projected_new_real/60:.2f}min)")
print(f"   • Gain estimé: {projected_old_real/projected_new_real:.1f}x plus rapide")
print(f"   • Temps économisé: {(projected_old_real - projected_new_real)/60:.1f}min")

print("\n" + "="*70)
print("✅ TEST TERMINÉ")
print("="*70)

ds.close()
