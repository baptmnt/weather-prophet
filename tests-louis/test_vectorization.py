"""
Test de performance pour mesurer les gains de la vectorisation par timestamp.
Compare l'ancien traitement ligne par ligne vs le nouveau traitement groupé.
"""

import time
import pandas as pd
import numpy as np
from pathlib import Path

print("="*70)
print("TEST D'OPTIMISATION - VECTORISATION PAR TIMESTAMP")
print("="*70)

# Simuler un DataFrame de stations avec timestamps répétés
print("\n📊 Génération de données de test...")

# Créer 1000 relevés avec 50 timestamps uniques (en moyenne 20 stations par timestamp)
np.random.seed(42)
n_samples = 1000
n_unique_timestamps = 50
n_stations = 100

timestamps = pd.date_range('2016-01-01', periods=n_unique_timestamps, freq='3H')
data = {
    'datetime': np.random.choice(timestamps, n_samples),
    'number_sta': np.random.choice(range(1000, 1000 + n_stations), n_samples),
    'lat': np.random.uniform(45, 47, n_samples),
    'lon': np.random.uniform(4, 6, n_samples),
    't': np.random.uniform(270, 290, n_samples),
    'hu': np.random.uniform(50, 100, n_samples),
}

stations_df = pd.DataFrame(data)
stations_df = stations_df.sort_values(['datetime', 'number_sta'])

print(f"   ✓ {len(stations_df)} relevés générés")
print(f"   ✓ {stations_df['datetime'].nunique()} timestamps uniques")
print(f"   ✓ {stations_df['number_sta'].nunique()} stations uniques")
print(f"   ✓ Moyenne: {len(stations_df) / stations_df['datetime'].nunique():.1f} stations par timestamp")

# Simuler le chargement d'images (avec délai artificiel)
def simulate_image_loading(timestamp):
    """Simule le temps de chargement d'un ensemble d'images satellites"""
    # Simuler 1ms de traitement (I/O, décompression, etc.)
    time.sleep(0.001)
    return {
        -12: {'IR108': np.random.rand(171, 261)},
        -24: {'IR108': np.random.rand(171, 261)},
    }

# TEST 1: Méthode ANCIENNE (ligne par ligne)
print("\n" + "="*70)
print("TEST 1: Méthode ANCIENNE (ligne par ligne)")
print("="*70)

start_old = time.time()
samples_old = []
image_loads_old = 0

for _, row in stations_df.iterrows():
    ref_time = pd.Timestamp(row['datetime'])
    
    # Charger les images (simulé)
    multi_images = simulate_image_loading(ref_time)
    image_loads_old += 1
    
    # Créer le sample
    sample = {
        'timestamp': ref_time,
        'station_id': int(row['number_sta']),
        'images': multi_images,
        't': row['t'],
        'hu': row['hu'],
    }
    samples_old.append(sample)

end_old = time.time()
elapsed_old = end_old - start_old

print(f"⏱️  Temps total: {elapsed_old:.3f}s")
print(f"📦 Samples créés: {len(samples_old)}")
print(f"🔄 Chargements d'images: {image_loads_old}")
print(f"⏱️  Temps moyen par sample: {elapsed_old/len(samples_old)*1000:.2f}ms")

# TEST 2: Méthode NOUVELLE (vectorisée par timestamp)
print("\n" + "="*70)
print("TEST 2: Méthode NOUVELLE (vectorisation par timestamp)")
print("="*70)

start_new = time.time()
samples_new = []
image_loads_new = 0
image_cache = {}

# Grouper par timestamp
grouped = stations_df.groupby('datetime')
print(f"   Groupes créés: {len(grouped)}")

for ref_time, group_df in grouped:
    ref_time = pd.Timestamp(ref_time)
    
    # Vérifier le cache
    cache_key = f"multi_{ref_time}"
    if cache_key in image_cache:
        multi_images = image_cache[cache_key]
    else:
        # Charger les images UNE SEULE FOIS
        multi_images = simulate_image_loading(ref_time)
        image_cache[cache_key] = multi_images
        image_loads_new += 1
    
    # Traiter TOUTES les stations de ce timestamp
    for _, row in group_df.iterrows():
        sample = {
            'timestamp': ref_time,
            'station_id': int(row['number_sta']),
            'images': multi_images,  # RÉUTILISATION
            't': row['t'],
            'hu': row['hu'],
        }
        samples_new.append(sample)

end_new = time.time()
elapsed_new = end_new - start_new

print(f"⏱️  Temps total: {elapsed_new:.3f}s")
print(f"📦 Samples créés: {len(samples_new)}")
print(f"🔄 Chargements d'images: {image_loads_new}")
print(f"⏱️  Temps moyen par sample: {elapsed_new/len(samples_new)*1000:.2f}ms")

# COMPARAISON
print("\n" + "="*70)
print("📊 RÉSULTATS")
print("="*70)

speedup = elapsed_old / elapsed_new
time_saved = elapsed_old - elapsed_new
reduction_io = (image_loads_old - image_loads_new) / image_loads_old * 100

print(f"\n🚀 Gain de performance:")
print(f"   • Méthode ancienne: {elapsed_old:.3f}s ({image_loads_old} chargements)")
print(f"   • Méthode nouvelle: {elapsed_new:.3f}s ({image_loads_new} chargements)")
print(f"   • Accélération: {speedup:.1f}x plus rapide")
print(f"   • Temps économisé: {time_saved:.3f}s")
print(f"   • Réduction I/O: -{reduction_io:.1f}% de chargements")

# Validation
print(f"\n✅ Validation:")
print(f"   • Nombre de samples: {len(samples_old)} vs {len(samples_new)} {'✓' if len(samples_old) == len(samples_new) else '✗'}")

# Projection pour 83,000 samples
print(f"\n🎯 PROJECTION POUR 83,000 SAMPLES:")

# Estimer le ratio timestamps uniques / total samples (ici ~20 stations/timestamp)
ratio = len(stations_df) / stations_df['datetime'].nunique()
estimated_unique_timestamps = int(83000 / ratio)

print(f"   • Ratio moyen: {ratio:.1f} stations par timestamp")
print(f"   • Timestamps uniques estimés: {estimated_unique_timestamps:,}")

projected_old = (elapsed_old / len(samples_old)) * 83000
projected_new = (elapsed_new / len(samples_new)) * 83000

print(f"   • Méthode ancienne: {projected_old:.1f}s ({projected_old/60:.1f}min)")
print(f"   • Méthode nouvelle: {projected_new:.1f}s ({projected_new/60:.1f}min)")
print(f"   • Temps économisé: {(projected_old - projected_new):.1f}s ({(projected_old - projected_new)/60:.1f}min)")

# Impact combiné avec étape 1
print(f"\n🎯 GAIN CUMULÉ (Étape 1 + Étape 2):")
print(f"   • Étape 1 (pré-indexation): 8-71x")
print(f"   • Étape 2 (vectorisation): {speedup:.1f}x")
print(f"   • Gain cumulé estimé: {8 * speedup:.0f}-{71 * speedup:.0f}x")

print("\n" + "="*70)
print("✅ TEST TERMINÉ")
print("="*70)
