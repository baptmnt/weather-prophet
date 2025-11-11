
# Weather Prophet 🌦️

Projet de prédiction météorologique par Machine Learning à partir de données satellites et de stations au sol.

## 🎯 Objectif du projet

Créer un modèle de ML capable de **prédire les conditions météorologiques au sol** (température, humidité, précipitations, etc.) à partir d'**images satellites historiques**.

### Principe

```text
Images satellites passées → Modèle ML → Prédiction météo au sol
(t-12h, t-1j, t-2j, t-7j)              (température, humidité, etc.)
```

---

## 📁 Structure du projet

```text
weather-prophet/
├── README.md                  # 📖 Ce fichier - documentation principale
├── tests-louis/               # 🧪 Scripts de traitement et ML
│   ├── create_ml_dataset.py       # 🔧 Génère le dataset HDF5 d'entraînement
│   ├── inspect_dataset.py         # 🔍 Inspecte et visualise le dataset
│   ├── pytorch_dataloader.py      # 🚀 DataLoader PyTorch pour le ML
│   ├── test_dataloader.py         # ✅ Test rapide du DataLoader
│   └── README_DATASET.md          # 📖 Documentation détaillée du dataset
├── datasets/                  # 📊 Datasets HDF5 générés
│   ├── meteonet_SE_2016_20160101.h5
│   └── *.png                  # Visualisations de samples
└── data_extraction/           # Scripts d'extraction (legacy)
```

---

## 🚀 Guide d'utilisation rapide

### Prérequis

```bash
# Installation des dépendances
pip install xarray h5netcdf h5py numpy pandas matplotlib torch torchvision imageio pillow
```

### 1️⃣ Générer le dataset ML

Le script `tests-louis/create_ml_dataset.py` combine les données satellites (.nc) et stations au sol (CSV) en un seul fichier HDF5 optimisé.

**✨ Optimisations v2.1 (Pré-indexation + Vectorisation)** :

- 🚀 **Étape 1 - Pré-indexation temporelle** : 8-71x plus rapide grâce à la recherche dichotomique O(log n)
- ⚡ **Étape 2 - Vectorisation par timestamp** : 14.6x plus rapide en groupant les requêtes
- 🎯 **Gain cumulé** : 117-1036x plus rapide que la version initiale
- 📊 Réduction de 95% des chargements d'images redondants
- 💾 Cache intelligent multi-niveaux pour réutilisation maximale

```bash
cd tests-louis

# Utilisation de base (avec chemins relatifs configurables)
python create_ml_dataset.py --data-root ../data --zone SE --year 2016

# Avec options avancées
python create_ml_dataset.py \
    --data-root ../data \
    --zone SE \
    --year 2016 \
    --output-dir ./datasets \
    --station-id 7149 \
    --save-intermediate \
    --chunk-size 500
```

**Arguments disponibles** :

- `--data-root` : Dossier racine contenant les zones (défaut : `data/`)
- `--zone` : Zone à traiter (`SE` ou `NW`, défaut : `SE`)
- `--year` : Année des fichiers satellites (défaut : `2016`)
- `--output-dir` : Dossier de sortie (défaut : `data/<ZONE>/datasets/`)
- `--station-id` : Filtrer sur une station spécifique (optionnel)
- `--save-intermediate` : Sauvegarder des chunks intermédiaires .npz
- `--chunk-size` : Taille des chunks (défaut : 500 samples)
- `--build-final` : Merger des chunks existants sans reconstruire
- `--merge-start` / `--merge-end` : Sélectionner la plage de chunks à merger
- `--intermediate-dir` : Dossier pour fichiers temporaires

**Configuration** (à modifier dans le script si nécessaire) :

```python
zone = 'SE'  # ou 'NW' (South-East ou North-West France)
year = 2016
```

**Optimisations de performance intégrées** :

1. **Pré-indexation temporelle** : Les timestamps sont indexés au chargement → recherche O(log n) au lieu de O(n)
2. **Recherche dichotomique** : Utilisation de `bisect` pour trouver les timestamps les plus proches
3. **Vectorisation par timestamp** : Groupement des stations par timestamp pour charger les images une seule fois
4. **Cache multi-niveaux** : Les images et ensembles multi-temporels sont mis en cache pour réutilisation maximale
5. **Réduction I/O** : 95% moins de lectures disque grâce au partage d'images entre stations

**Sortie** :

- `datasets/meteonet_SE_2016.h5` (~740 MB par jour)
- Logs de progression et statistiques de construction

**Ce que fait le script** :

1. ✅ Charge les fichiers satellites NetCDF (CT, IR039, IR108, VIS06, WV062) avec **indexation temporelle**
2. ✅ Charge les mesures des stations au sol depuis le CSV avec **pré-indexation (station, timestamp)**
3. ✅ **Groupement intelligent** : Traite les stations par batch de timestamps identiques
4. ✅ Pour chaque timestamp unique :
    - Extrait les **images satellites complètes** à t-12h, t-24h, t-48h, t-168h **une seule fois** via **recherche dichotomique O(log n)**
    - Réutilise ces images pour **toutes les stations** du même timestamp
    - Récupère les mesures au sol (t, hu, precip, dd, ff, psl, td) pour chaque station
    - Aligne temporellement et spatialement les données
5. ✅ Sauvegarde en format HDF5 compressé avec metadata (ou chunks intermédiaires .npz)

**Temps d'exécution** :

- ⚡ **v2.1 optimisé** : ~0.5-2 secondes pour 1 jour de données (117-1036x plus rapide)
- 📦 **Mode chunks** : Traite par blocs de 500 samples pour éviter la saturation mémoire
- 🎯 **83,000 samples** : Estimé à ~12-30 secondes au lieu de plusieurs heures

---

### 2️⃣ Inspecter le dataset

Le script `tests-louis/inspect_dataset.py` permet de vérifier la qualité et la structure du dataset généré.

```bash
cd tests-louis
python inspect_dataset.py [chemin_dataset.h5]
```

Si aucun chemin n'est fourni, il inspecte automatiquement le dernier dataset créé.

**Ce qu'il affiche** :

- 📋 Structure du fichier HDF5 (dimensions, attributs)
- 📊 Statistiques sur les images et labels (min, max, mean, NaN%)
- 📍 Info spatiales (stations, coordonnées)
- ⏰ Info temporelles (période couverte)
- 🔍 Qualité des données (complétude par canal et timestep)
- 🎨 Visualisations de samples aléatoires (PNG générés)

**Sortie** :

- Statistiques dans le terminal
- `datasets/sample_XXX_visualization.png` (3 samples aléatoires)

---

### 3️⃣ Charger le dataset avec PyTorch

Le script `tests-louis/pytorch_dataloader.py` fournit un DataLoader prêt à l'emploi pour l'entraînement.

#### Utilisation simple

```python
from tests-louis.pytorch_dataloader import create_dataloaders
from pathlib import Path

# Créer les DataLoaders (train/val/test splits automatiques)
train_loader, val_loader, test_loader = create_dataloaders(
    Path("datasets/meteonet_SE_2016_20160101.h5"),
    batch_size=32,
    train_split=0.7,    # 70% train
    val_split=0.15,     # 15% validation
    num_workers=4       # Chargement parallèle
)

# Itérer sur les batchs
for images, labels, metadata in train_loader:
    # images: (batch_size, 4_timesteps, 5_channels, height, width)
    # labels: (batch_size, 7_variables)
    # metadata: dict avec timestamp, station_id, coords, etc.

    # → Votre modèle ici !
    predictions = model(images)
    loss = criterion(predictions, labels)
    # ...
```

#### Utilisation avancée

```python
from tests-louis.pytorch_dataloader import MeteoNetDataset

# Dataset custom avec options
dataset = MeteoNetDataset(
    h5_path,
    normalize=True,        # Normalisation Z-score par canal
    handle_nans='zero',    # Options: 'zero', 'mean', 'keep'
    transform=None,        # Transformations custom (augmentation)
)

# Accéder aux infos du dataset
print(dataset.get_channel_names())  # ['CT', 'IR039', 'IR108', 'VIS06', 'WV062']
print(dataset.get_target_names())   # ['dd', 'ff', 'precip', 'hu', 'td', 't', 'psl']
print(dataset.get_timesteps())      # [-12, -24, -48, -168]
```

---

### 4️⃣ Tester le DataLoader

Script de test rapide pour vérifier que tout fonctionne.

```bash
cd tests-louis
python test_dataloader.py
```

**Ce qu'il fait** :

- ✅ Charge le dataset
- ✅ Crée les DataLoaders
- ✅ Teste le chargement d'un batch
- ✅ Affiche les shapes et statistiques

**Sortie attendue** :

```text
✅ TOUT FONCTIONNE PARFAITEMENT!

🚀 Vous pouvez maintenant:
    1. Créer un modèle PyTorch
    2. Itérer sur train_loader pour l'entraînement
    3. Évaluer sur val_loader et test_loader
```

---

## 📊 Format du dataset

### Structure HDF5

```text
dataset.h5
├── images/                    (N, 4, 5, 171, 261)
│   └── Images satellites :
│       - Dimension 0: Samples
│       - Dimension 1: Timesteps [-12h, -24h, -48h, -168h]
│       - Dimension 2: Canaux [CT, IR039, IR108, VIS06, WV062]
│       - Dimension 3-4: Spatial (171×261 pixels, ~3km/pixel)
│
├── labels/                    (N, 7)
│   └── Variables météo au sol :
│       [dd, ff, precip, hu, td, t, psl]
│
└── metadata/
    ├── timestamps         Timestamp de chaque sample
    ├── station_ids        ID de la station
    ├── station_coords     Coordonnées (lat, lon)
    ├── station_heights    Altitude (m)
    └── zones             Zone géographique ('NW' ou 'SE')
```

### Variables

**Canaux satellites** :

- `CT` : Type de nuage (0-15, catégoriel)
- `IR039` : Infrarouge 3.9 µm (°C)
- `IR108` : Infrarouge 10.8 µm (°C)
- `VIS06` : Visible 0.6 µm (%, jour uniquement)
- `WV062` : Vapeur d'eau 6.2 µm (°C)

**Labels stations** :

- `dd` : Direction du vent (°)
- `ff` : Vitesse du vent (m.s^-1^)
- `precip` : Précipitations (kg.m^2^)
- `hu` : Humidité (%)
- `td` : Température du point de rosée (K)
- `t` : Température (K)
- `psl` : Pression au niveau de la mer (Pa)

---

## 🔧 Configuration et personnalisation

### Modifier les timesteps

Dans `tests-louis/create_ml_dataset.py` :

```python
class Config:
    TIMESTEPS = [-6, -12, -24, -72]  # 6h, 12h, 1j, 3j au lieu de [-12, -24, -48, -168]
```

### Changer les canaux satellites

```python
class Config:
    CHANNELS = ['IR108', 'VIS06', 'WV062']  # Seulement 3 canaux
```

### Sélectionner certaines variables cibles

```python
class Config:
    TARGET_VARS = ['t', 'hu', 'precip']  # Seulement température, humidité, précipitations
```

### Ajuster la normalisation

```python
dataset = MeteoNetDataset(
    h5_path,
    normalize=True,      # True = normalisation Z-score
    handle_nans='zero'   # 'zero' | 'mean' | 'keep'
)
```

---

## 📈 Statistiques du dataset (exemple SE_20160101)

### Données générales

- **Samples** : 2902
- **Stations uniques** : 335
- **Dimensions images** : 171 × 261 pixels
- **Canaux disponibles** : 4 (IR039, IR108, VIS06, WV062) - CT absent pour SE
- **Taux de réussite** : 60% (samples avec images ET labels valides)

### Qualité des labels

| Variable | Couverture | Min     | Max      | Mean    |
|----------|-----------|---------|----------|---------|
| t        | 99.9%     | 264 K   | 288 K    | 281 K   |
| hu       | 97.9%     | 44%     | 106%     | 88%     |
| precip   | 96.9%     | 0 kg/m² | 0.8 kg/m²| 0.02    |
| dd       | 93.5%     | 0°      | 360°     | 138°    |
| ff       | 93.5%     | 0 m/s   | 13.7 m/s | 3.4 m/s |
| td       | 97.6%     | 263 K   | 286 K    | 279 K   |
| psl      | 20.0%     | 101190 Pa | 102270 Pa | 101785 Pa |

**Note** : `psl` a peu de couverture (beaucoup de stations ne mesurent pas cette variable).

---

## 🧠 Architecture recommandée pour le ML

### Pourquoi des images complètes ?

✅ **Contexte spatial** : Le modèle voit les systèmes météo qui s'approchent  
✅ **Dynamique temporelle** : Apprend la vitesse/direction des mouvements  
✅ **Généralisation** : Comprend la relation "météo régionale → météo locale"  
✅ **Efficacité** : Une image sert pour toutes les stations de la zone

### Timesteps choisis

- **t-12h** : Tendance court terme
- **t-24h** : Évolution sur 1 jour
- **t-48h** : Dynamique à 2 jours
- **t-168h** : Contexte hebdomadaire

### Suggestions d'architectures

1. **CNN 3D** : Pour traiter séquences spatiotemporelles
2. **ConvLSTM** : Combine convolutions + mémoire temporelle
3. **U-Net temporel** : Si vous voulez faire de la prédiction spatiale complète
4. **Vision Transformer** : Pour capturer dépendances long-terme

---

## 📚 Documentation complète

Pour plus de détails sur la structure HDF5, l'utilisation avancée, le troubleshooting, etc. :

👉 **Voir `tests-louis/README_DATASET.md`**

---

## ⚠️ Points d'attention

### Valeurs manquantes (NaN)

- **VIS06** : Pas de données la nuit (normal)
- **CT** : Peut être absent selon la zone
- **Labels** : `psl` souvent absent (80% de NaN)

**Recommandation** : Utiliser `handle_nans='zero'` dans le DataLoader.

### Performance

- **Dataset complet** : ~740 MB pour 1 jour → ~270 GB pour 1 an
- **Solution** : HDF5 permet le chargement lazy (pas tout en RAM)
- **Optimisation** : Augmenter `num_workers` dans le DataLoader

### Gestion mémoire

```python
# Pour gros datasets, réduire batch_size ou utiliser gradient accumulation
train_loader = DataLoader(dataset, batch_size=16)  # Au lieu de 32
```

---

## 🚀 Prochaines étapes

### Pour commencer l'entraînement

1. **Créer un modèle simple** pour tester le pipeline
2. **Définir la loss function** (MSE, MAE, ou custom ignorant les NaN)
3. **Entraîner** sur quelques epochs
4. **Évaluer** les performances sur val/test

### Pour étendre le dataset

1. **Dataset complet** : Générer pour toute l'année 2016
2. **Multi-zones** : Combiner NW et SE
3. **Autres sources** : Ajouter données radar, AROME/ARPEGE
4. **Multi-années** : 2016, 2017, 2018...

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
```

### Performances lentes

- Augmenter `num_workers` dans DataLoader
- Vérifier que le HDF5 est sur un SSD
- Réduire `batch_size` si RAM insuffisante

### Dataset trop gros

- Réduire la période (1 semaine au lieu de 1 mois)
- Sous-échantillonner temporellement (1 sample toutes les 3h)
- Utiliser moins de canaux satellites

---

## 📞 Contact et contribution

Projet développé dans le cadre du TIP - INSA Lyon 4TCA.

**Sources de données** : [MeteoNet](https://meteonet.umr-cnrm.fr/) - Météo-France

---

## 📄 Licence

Voir `LICENCE.md` pour les détails.

---

Bon entraînement ! 🌦️🚀
