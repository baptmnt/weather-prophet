# Dataset ML MeteoNet - Guide d'utilisation

## 📋 Vue d'ensemble

Ce projet génère un dataset ML au format HDF5 pour la prédiction météorologique à partir de données satellites et de stations au sol.

### Objectif
Entraîner un modèle de ML qui, à partir d'images satellites historiques (t-12h, t-1j, t-2j, t-7j), prédit les mesures météo au sol au temps t (température, humidité, précipitations, etc.).

---

## 🗂️ Structure du dataset HDF5

```
dataset.h5
├── images/                    (N, 4, 5, 171, 261)
│   └── Images satellites complètes
│       - Dimension 0: Samples (N)
│       - Dimension 1: Timesteps (4) → [-12h, -24h, -48h, -168h]
│       - Dimension 2: Canaux (5) → [CT, IR039, IR108, VIS06, WV062]
│       - Dimension 3-4: Spatial (171×261 pixels)
│
├── labels/                    (N, 7)
│   └── Mesures stations au sol
│       - dd: direction du vent (°)
│       - ff: vitesse du vent (m/s)
│       - precip: précipitations (kg/m²)
│       - hu: humidité (%)
│       - td: température du point de rosée (K)
│       - t: température (K)
│       - psl: pression au niveau de la mer (Pa)
│
└── metadata/
    ├── timestamps         (N,)  - Timestamp de chaque sample
    ├── station_ids        (N,)  - ID de la station
    ├── station_coords     (N,2) - Coordonnées (lat, lon)
    ├── station_heights    (N,)  - Altitude de la station (m)
    └── zones             (N,)   - Zone géographique ('NW' ou 'SE')
```

### Attributs du fichier
- `n_samples`: Nombre total de samples
- `n_timesteps`: Nombre de pas de temps passés (4)
- `n_channels`: Nombre de canaux satellites (5)
- `n_labels`: Nombre de variables météo (7)
- `image_height`, `image_width`: Dimensions spatiales
- `channels`: Liste des canaux
- `target_vars`: Liste des variables cibles
- `timesteps`: Liste des offsets temporels
- `creation_date`: Date de création

---

## 🚀 Utilisation

### 1. Génération du dataset

```bash
python create_ml_dataset.py
```

**Configuration** (dans le script) :
```python
zone = 'SE'  # ou 'NW'
year = 2016
date = '20160101'  # Date du CSV des stations
```

**Sortie** : `datasets/meteonet_SE_2016_20160101.h5` (~740 MB avec compression)

### 2. Inspection du dataset

```bash
python inspect_dataset.py [chemin_vers_dataset.h5]
```

Affiche :
- ✅ Structure et dimensions
- ✅ Statistiques sur les images et labels
- ✅ Qualité des données (taux de NaN, complétude)
- ✅ Visualisations de samples aléatoires

### 3. Chargement avec PyTorch

```python
from pytorch_dataloader import MeteoNetDataset, create_dataloaders
from pathlib import Path

# Chemin du dataset
dataset_path = Path("datasets/meteonet_SE_2016_20160101.h5")

# Créer les DataLoaders (train/val/test splits)
train_loader, val_loader, test_loader = create_dataloaders(
    dataset_path,
    batch_size=32,
    train_split=0.7,
    val_split=0.15,
    num_workers=4
)

# Itérer sur les batchs
for images, labels, metadata in train_loader:
    # images: (batch, 4_timesteps, 5_channels, 171, 261)
    # labels: (batch, 7)
    # metadata: dict avec timestamp, station_id, coords, etc.
    
    # Votre code d'entraînement ici
    pass
```

---

## 📊 Statistiques du dataset (sample SE_20160101)

### Données générales
- **Samples totaux** : 2902
- **Stations uniques** : 335
- **Zone couverte** : Sud-Est France
- **Latitude** : 41.37° - 46.23°
- **Longitude** : 2.00° - 9.54°

### Images satellites
- **Dimensions** : 171 × 261 pixels (~3 km/pixel)
- **Canaux disponibles** : IR039, IR108, VIS06, WV062 (CT absent pour SE)
- **Résolution temporelle** : 1 heure
- **NaN ratio** : ~50% (normal, VIS06 n'a pas de données de nuit)

### Labels stations
| Variable | Min     | Max      | Mean    | NaN% |
|----------|---------|----------|---------|------|
| dd       | 0°      | 360°     | 138°    | 6.5% |
| ff       | 0 m/s   | 13.7 m/s | 3.4 m/s | 6.5% |
| precip   | 0 kg/m² | 0.8 kg/m²| 0.02    | 3.1% |
| hu       | 44%     | 106%     | 88%     | 2.1% |
| td       | 263 K   | 286 K    | 279 K   | 2.4% |
| t        | 264 K   | 288 K    | 281 K   | 0.1% |
| psl      | 101190 Pa | 102270 Pa | 101785 Pa | 80.0% |

**Note** : `psl` a beaucoup de NaN (80%) car peu de stations mesurent cette variable.

---

## 🧠 Architecture du dataset pour le ML

### Pourquoi des images complètes ?

✅ **Contexte spatial** : Le modèle voit les systèmes météo régionaux qui s'approchent de la station

✅ **Dynamique temporelle** : Avec 4 timesteps, le modèle apprend la vitesse et direction des mouvements

✅ **Généralisation** : Le modèle apprend la relation "dynamique régionale → météo locale" plutôt que des patterns spécifiques à une position

✅ **Efficacité** : Une image peut servir pour toutes les stations de la zone (pas de duplication)

### Timesteps choisis
- **t-12h** : Météo récente (tendance à court terme)
- **t-24h** : Évolution sur 1 jour
- **t-48h** : Dynamique à 2 jours
- **t-168h** : Contexte à 1 semaine (patterns saisonniers)

---

## 🔧 Personnalisation

### Modifier les timesteps

Dans `create_ml_dataset.py` :
```python
class Config:
    TIMESTEPS = [-6, -12, -24, -72]  # Exemple : 6h, 12h, 1j, 3j
```

### Ajouter/retirer des canaux

```python
class Config:
    CHANNELS = ['IR108', 'VIS06', 'WV062']  # Exemple : seulement 3 canaux
```

### Changer les variables cibles

```python
class Config:
    TARGET_VARS = ['t', 'hu', 'precip']  # Exemple : seulement 3 variables
```

### Ajuster la normalisation

Dans `pytorch_dataloader.py` :
```python
dataset = MeteoNetDataset(
    h5_path,
    normalize=True,      # Normalisation Z-score
    handle_nans='zero'   # Options: 'zero', 'mean', 'keep'
)
```

---

## 📝 Format des données

### Images satellites

| Canal  | Type    | Unité | Description                    | Fréquence |
|--------|---------|-------|--------------------------------|-----------|
| CT     | uint8   | -     | Type de nuage (0-15)           | 15 min    |
| IR039  | float32 | °C    | Infrarouge 3.9 µm              | 1 heure   |
| IR108  | float32 | °C    | Infrarouge 10.8 µm             | 1 heure   |
| VIS06  | float32 | %     | Visible 0.6 µm (jour seul)     | 1 heure   |
| WV062  | float32 | °C    | Vapeur d'eau 6.2 µm            | 1 heure   |

### Labels stations

Toutes les variables suivent le format du CSV MeteoNet (voir `content.md`).

---

## ⚠️ Limitations et considérations

### Valeurs manquantes (NaN)

1. **VIS06** : Pas de données de nuit (normal)
2. **CT** : Peut être absent selon la zone
3. **Labels** : Certaines stations ne mesurent pas toutes les variables
4. **Timesteps** : Si données satellites manquantes à un timestep

**Gestion recommandée** :
- Images : Remplacer NaN par 0 ou moyenne du canal
- Labels : Filtrer les samples avec trop de NaN, ou utiliser des loss functions robustes

### Taille du dataset

- **Sample (1 jour, 1 zone)** : ~740 MB compressé
- **Année complète** : ~270 GB compressé (estimé)
- **Multi-années** : Peut nécessiter plusieurs fichiers HDF5

**Solution** : Le format HDF5 permet le chargement lazy (pas tout en RAM).

### Performance

- **Lecture HDF5** : Très rapide avec chunks adaptés
- **DataLoader PyTorch** : Utiliser `num_workers > 0` pour paralléliser
- **Cache** : Les statistiques de normalisation sont calculées une fois au chargement

---

## 🎯 Prochaines étapes

### Pour l'entraînement ML

1. **Créer un modèle** : CNN, U-Net, ou Transformer pour traiter les séquences d'images
2. **Définir la loss** : MSE pour régression, ou loss custom ignorant les NaN
3. **Augmentation de données** : Rotations, flips, crops (attention à la cohérence temporelle)
4. **Validation** : Comparer prédictions vs mesures réelles

### Pour étendre le dataset

1. **Dataset complet** : Générer pour toute l'année 2016
2. **Multi-zones** : Combiner NW et SE
3. **Données radar** : Ajouter les précipitations radar comme canal supplémentaire
4. **Données AROME/ARPEGE** : Ajouter les sorties de modèles numériques

---

## 📚 Références

- **MeteoNet** : [Dataset météo de Météo-France](https://meteonet.umr-cnrm.fr/)
- **HDF5** : [Format de données scientifiques](https://www.hdfgroup.org/)
- **PyTorch Dataset** : [Documentation officielle](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)

---

## 💡 Astuces

### Accès rapide aux attributs

```python
import h5py

with h5py.File('dataset.h5', 'r') as f:
    print(f"Samples: {f.attrs['n_samples']}")
    print(f"Canaux: {f.attrs['channels']}")
```

### Charger un sample spécifique

```python
with h5py.File('dataset.h5', 'r') as f:
    sample_100_images = f['images'][100]
    sample_100_labels = f['labels'][100]
```

### Filtrer par station

```python
with h5py.File('dataset.h5', 'r') as f:
    station_ids = f['metadata/station_ids'][:]
    station_123_indices = np.where(station_ids == 1234567)[0]
    station_123_images = f['images'][station_123_indices]
```

---

## 🐛 Troubleshooting

### Erreur "No module named 'h5py'"
```bash
pip install h5py
```

### Erreur "Indexing elements must be in increasing order"
HDF5 nécessite des indices triés :
```python
indices = np.sort(indices)
data = dataset[indices]
```

### Performances lentes
- Augmenter `num_workers` dans le DataLoader
- Vérifier que le fichier HDF5 est sur un SSD
- Réduire `batch_size` si RAM insuffisante

### NaN dans les prédictions
- Vérifier `handle_nans='zero'` dans le dataset
- Utiliser une loss function robuste aux NaN
- Filtrer les samples avec trop de valeurs manquantes

---

**Bon entraînement ! 🚀**
