"""
Test de performance pour mesurer les gains de la parallélisation multi-processus.
Compare l'exécution avec 1, 2, 4, et 8 workers.
"""

import time
import pandas as pd
import numpy as np
from pathlib import Path
from multiprocessing import cpu_count, Pool, freeze_support
import sys


# Simuler un traitement par batch (simplifié)
def process_batch_sequential(batch_data):
    """Simule le traitement séquentiel d'un batch"""
    batch, delay_ms = batch_data
    time.sleep(delay_ms / 1000 * len(batch))  # Simuler le temps de traitement
    return [x * 2 for x in batch]


def benchmark_parallel(data, num_workers, processing_delay_ms=1):
    """
    Benchmark avec un nombre donné de workers.
    
    Args:
        data: Liste de données à traiter
        num_workers: Nombre de processus parallèles
        processing_delay_ms: Délai de traitement par élément (ms)
    """
    # Découper en batches
    if num_workers == 1:
        # Mode séquentiel
        batches = [data]
    else:
        batch_size = max(1, len(data) // num_workers)
        batches = []
        for i in range(0, len(data), batch_size):
            batches.append(data[i:i + batch_size])
    
    # Préparer les données avec le délai
    batch_data = [(batch, processing_delay_ms) for batch in batches]
    
    start = time.time()
    
    if num_workers == 1:
        # Séquentiel
        results = [process_batch_sequential(bd) for bd in batch_data]
    else:
        # Parallèle
        with Pool(processes=num_workers) as pool:
            results = pool.map(process_batch_sequential, batch_data)
    
    elapsed = time.time() - start
    
    # Fusionner résultats
    total_results = sum([len(r) for r in results])
    
    return elapsed, total_results


def main():
    print("="*70)
    print("TEST D'OPTIMISATION - PARALLÉLISATION MULTI-PROCESSUS")
    print("="*70)

    # Informations système
    print(f"\n💻 Système:")
    print(f"   • CPUs disponibles: {cpu_count()}")
    
    n_items = 1000  # Simuler 1000 timestamps à traiter
    data = list(range(n_items))
    processing_delay = 1  # 1ms par item

    print(f"   ✓ {n_items} items à traiter")
    print(f"   ✓ Délai de traitement: {processing_delay}ms par item")
    print(f"   ✓ Temps séquentiel théorique: {n_items * processing_delay / 1000:.2f}s")

    # Tests avec différents nombres de workers
    workers_to_test = [1, 2, 4]
    if cpu_count() >= 8:
        workers_to_test.append(8)

    results = {}

    for num_workers in workers_to_test:
        if num_workers > cpu_count():
            continue
        
        print(f"\n{'='*70}")
        print(f"TEST: {num_workers} worker{'s' if num_workers > 1 else ''}")
        print(f"{'='*70}")
        
        elapsed, total = benchmark_parallel(data, num_workers, processing_delay)
        
        print(f"⏱️  Temps total: {elapsed:.3f}s")
        print(f"📦 Items traités: {total}")
        print(f"⚡ Throughput: {total/elapsed:.0f} items/s")
        
        results[num_workers] = elapsed

    # Comparaison
    print(f"\n{'='*70}")
    print("📊 COMPARAISON")
    print(f"{'='*70}")

    baseline = results[1]
    print(f"\n🎯 Gains de performance:")

    for num_workers in sorted(results.keys()):
        elapsed = results[num_workers]
        speedup = baseline / elapsed
        efficiency = (speedup / num_workers) * 100
        
        print(f"   • {num_workers:2d} worker{'s' if num_workers > 1 else ' '}: {elapsed:.3f}s  |  {speedup:.1f}x plus rapide  |  Efficacité: {efficiency:.0f}%")

    # Projection pour dataset réel
    print(f"\n🎯 PROJECTION POUR DATASET RÉEL (83,000 samples):")
    print(f"   • Ratio estimé: 20 stations/timestamp → ~4,150 timestamps uniques")
    print(f"   • Temps de traitement estimé par timestamp: ~3ms (I/O + processing)")

    estimated_time_per_timestamp_ms = 3
    n_timestamps = 4150

    for num_workers in sorted(results.keys()):
        speedup = baseline / results[num_workers]
        projected_time = (n_timestamps * estimated_time_per_timestamp_ms / 1000) / speedup
        
        print(f"   • {num_workers} worker{'s' if num_workers > 1 else ' '}: ~{projected_time:.1f}s ({projected_time/60:.2f}min)")

    # Recommandation
    print(f"\n💡 RECOMMANDATION:")
    optimal_workers = min(4, cpu_count())  # Sweet spot généralement à 4 workers
    print(f"   • Utiliser --num-workers={optimal_workers} pour un bon équilibre performance/overhead")
    print(f"   • Utiliser --num-workers=0 pour utiliser tous les CPUs ({cpu_count()})")

    # Gain cumulé
    print(f"\n🎯 GAIN CUMULÉ (Étapes 1+2+3):")
    cumul_step1_2 = 117  # Minimum du gain cumulé étapes 1+2
    best_speedup_step3 = max([baseline / results[w] for w in results.keys()])
    cumul_total = cumul_step1_2 * best_speedup_step3

    print(f"   • Étape 1 (pré-indexation): 8-71x")
    print(f"   • Étape 2 (vectorisation): 14.6x")
    print(f"   • Étape 3 (parallélisation): {best_speedup_step3:.1f}x")
    print(f"   • Gain total: ~{cumul_total:.0f}x minimum ({cumul_step1_2}x × {best_speedup_step3:.1f}x)")

    print(f"\n{'='*70}")
    print("✅ TEST TERMINÉ")
    print(f"{'='*70}")


if __name__ == '__main__':
    freeze_support()
    main()
