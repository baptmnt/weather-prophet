# 📊 Explication détaillée du fichier NetCDF CT_NW_2016.nc

## 🎯 Qu'est-ce qu'un fichier NetCDF (.nc) ?

**NetCDF** (Network Common Data Form) est un format de fichier autodescriptif pour stocker des données scientifiques multidimensionnelles. Il contient :
- Les **données** (valeurs numériques)
- Les **métadonnées** (descriptions, unités, coordonnées)
- Les **dimensions** (temps, latitude, longitude, etc.)

## 📦 Contenu exact de CT_NW_2016.nc

### Structure des données

```
Tableau 3D : (9, 188, 261)
           │   │    └─► 261 points de longitude
           │   └──────► 188 points de latitude  
           └──────────► 9 pas de temps
```

**Taille totale** : 9 × 188 × 261 = **441,828 pixels** = 431.3 KB

### 📍 Coordonnées spatiales

| Dimension | Plage | Résolution | Couverture |
|-----------|-------|------------|------------|
| **Latitude** | 46.25° → 51.90° N | 0.03° (~3 km) | Centre France → Belgique |
| **Longitude** | -5.84° → 2.00° E | 0.03° (~3 km) | Atlantique → Est France |

**Zone couverte** : Nord-Ouest de la France (NW)
- Inclut : Bretagne, Normandie, Pays de la Loire, région parisienne
- Surface : ~565 km (lat) × ~550 km (lon)

### ⏰ Coordonnées temporelles

**9 instants** espacés de 15 minutes :
```
[0] 2016-01-01 00:00:00
[1] 2016-01-01 00:15:00
[2] 2016-01-01 00:30:00
[3] 2016-01-01 00:45:00
[4] 2016-01-01 01:00:00
[5] 2016-01-01 01:30:00  ⚠️ Gap de 30 min
[6] 2016-01-01 01:45:00
[7] 2016-01-01 02:00:00
[8] 2016-01-01 02:15:00
```

**Durée totale** : 2h15 (première nuit du 1er janvier 2016)

### 🌥️ Variable CT (Cloud Type)

**Type de données** : `uint8` (entier non signé 8 bits)
**Valeurs possibles** : 0 à 15 (16 catégories)

#### Classification complète des types de nuages

| Valeur | Catégorie | Description | Présent ? |
|--------|-----------|-------------|-----------|
| 0 | No data | Pas de données | ❌ |
| 1 | Cloud-free land | Terre sans nuages | ✅ (32.8%) |
| 2 | Cloud-free sea | Mer sans nuages | ✅ (17.1%) |
| 3 | Snow over land | Neige sur terre | ❌ |
| 4 | Sea ice | Glace de mer | ❌ |
| 5 | Very low clouds | Nuages très bas (brouillard, stratus) | ✅ (9.3%) |
| 6 | Low clouds | Nuages bas (cumulus, stratocumulus) | ✅ (1.0%) |
| 7 | Mid-level clouds | Nuages moyens (altocumulus) | ✅ (0.02%) |
| 8 | High opaque clouds | Nuages hauts opaques (cumulonimbus) | ✅ (0.2%) |
| 9 | Very high opaque | Nuages très hauts opaques | ✅ (0%) |
| 10 | Fractional clouds | Nuages fragmentés/fractals | ✅ (19.2%) |
| 11 | High semitransp. thin | Nuages hauts semi-transparents fins (cirrus) | ✅ (11.7%) |
| 12 | High semitransp. medium | Nuages hauts semi-transp. moyens (cirrostratus) | ✅ (7.6%) |
| 13 | High semitransp. thick | Nuages hauts semi-transp. épais | ✅ (1.1%) |
| 14 | High + low/medium | Nuages hauts au-dessus de bas/moyens | ❌ |
| 15 | High + snow/ice | Nuages hauts au-dessus neige/glace | ❌ |

## 🖼️ Interprétation de l'image générée

### Structure de l'image

```
┌─────────────────┬─────────────────┐
│   00:00:00      │   00:30:00      │  ← Haut : début de période
├─────────────────┼─────────────────┤
│   01:30:00      │   02:15:00      │  ← Bas : fin de période
└─────────────────┴─────────────────┘
```

### Code couleur (palette automatique matplotlib)

| Couleur | Valeur CT | Signification |
|---------|-----------|---------------|
| **🟣 VIOLET FONCÉ** | 11-13 | Nuages hauts semi-transparents (cirrus/cirrostratus) |
| **🟢 VERT** | 5-6, 10 | Nuages bas/très bas, nuages fragmentés |
| **🟡 JAUNE** | 1-2 | Ciel dégagé (terre et mer) |
| **🔵 BLEU/CYAN** | 8-9 | Nuages hauts opaques |

### Interprétation météorologique

#### Zone JAUNE (Sud-Ouest, ~latitude 47-48°)
- **Signification** : Ciel dégagé sur l'océan Atlantique et le sud-ouest de la France
- **Surface** : ~50% de la zone (1-2)
- **Stabilité** : Zone stable sur toute la période

#### Zone VERTE (Centre, bandes horizontales)
- **Signification** : Nuages bas, brouillard, stratus
- **Altitude** : < 2000m
- **Risque** : Visibilité réduite au sol

#### Zone VIOLETTE (Nord et Est dominant)
- **Signification** : Système nuageux d'altitude (cirrus, cirrostratus)
- **Altitude** : > 6000m
- **Interprétation** : Front chaud ou perturbation en approche
- **Particularité** : Semi-transparent → visible par satellite infrarouge

### Évolution temporelle observable

```
00:00 → 00:30 : Stabilité relative, légère progression du système haut
00:30 → 01:30 : Extension des nuages bas vers le centre
01:30 → 02:15 : Développement marqué de nuages bas au sud
```

**Dynamique générale** :
- Maintien de la zone dégagée au sud-ouest
- Évolution lente du système d'altitude (violet)
- Variabilité rapide des nuages bas (vert)

## 🔬 Utilisations scientifiques

### 1. Nowcasting (prévision 0-3h)
- Suivi en temps réel des formations nuageuses
- Prédiction de l'évolution à très court terme
- Détection de systèmes convectifs

### 2. Analyse synoptique
- Identification de fronts météorologiques
- Caractérisation de masses d'air
- Suivi de perturbations

### 3. Validation de modèles
- Comparaison avec sorties de modèles numériques (AROME, ARPEGE)
- Évaluation de la qualité des prévisions
- Ajustement des paramétrisations

### 4. Machine Learning
- **Input** pour modèles de prévision
- Classification automatique de types de temps
- Prédiction de précipitations

### 5. Climatologie
- Statistiques sur couverture nuageuse
- Étude de la variabilité saisonnière
- Analyse de tendances

## 💾 Format technique des données

### Structure interne du fichier

```
CT_NW_2016.nc
├── Dimensions
│   ├── time: 9
│   ├── lat: 188
│   └── lon: 261
│
├── Coordonnées
│   ├── time(time): datetime64[ns]
│   ├── lat(lat): float64 [46.25 → 51.90]
│   └── lon(lon): float64 [-5.84 → 2.00]
│
├── Variables
│   └── CT(time, lat, lon): uint8
│
└── Attributs globaux
    ├── creating_function: "create_nc_file"
    └── appending_function: "append_unlimited_dim_nc_file"
```

### Accès aux données (Python)

```python
import xarray as xr

# Charger le fichier (IMPORTANT: engine='h5netcdf' pour Python 3.13)
data = xr.open_dataset("CT_NW_2016.nc", engine='h5netcdf')

# Accéder à la variable CT
ct = data['CT']  # xarray.DataArray (9, 188, 261)

# Sélectionner un instant
ct_t0 = ct.isel(time=0)  # Premier instant (00:00)
ct_date = ct.sel(time='2016-01-01T01:00')  # Par date

# Extraire les valeurs numpy
values = ct.values  # numpy.ndarray uint8

# Coordonnées
times = data.time.values  # array de datetime64
lats = data.lat.values    # array de float64
lons = data.lon.values    # array de float64
```

## 📈 Statistiques (premier instant, 00:00)

| Type de nuage | Code | Pixels | % |
|---------------|------|--------|---|
| Terre dégagée | 1 | 16,084 | 32.78% |
| Mer dégagée | 2 | 8,373 | 17.06% |
| Nuages fractals | 10 | 9,398 | 19.15% |
| Cirrus fins | 11 | 5,735 | 11.69% |
| Nuages très bas | 5 | 4,582 | 9.34% |
| Cirrostratus moyens | 12 | 3,734 | 7.61% |
| Cirrostratus épais | 13 | 551 | 1.12% |
| Nuages bas | 6 | 504 | 1.03% |

**Total ciel dégagé** : 49.84% (terre + mer)
**Total nuages hauts** : 20.42% (codes 11-13)
**Total nuages bas** : 10.37% (codes 5-6)

## 🌍 Contexte géographique

**Zone NW (Nord-Ouest)** couvre approximativement :
- **Régions françaises** : Bretagne, Normandie, Pays de la Loire, Hauts-de-France (partie), Île-de-France
- **Pays voisins** : Sud de l'Angleterre, Belgique (partie)
- **Océan** : Manche, partie de l'Atlantique

**Particularités climatiques** :
- Influence océanique forte
- Variabilité rapide
- Fréquence élevée de nuages bas
- Passages frontaux réguliers

---

## 🎓 Points clés à retenir

1. **Fichier NetCDF** = conteneur autodescriptif pour données géospatiales 3D
2. **CT** = Classification de types de nuages en 16 catégories
3. **Résolution** = ~3 km au sol, 15 minutes en temps
4. **Usage** = Nowcasting, validation modèles, machine learning
5. **Image** = Visualisation de 4 instants sur 2h15 d'évolution nuageuse
6. **Couleurs** = Mapping direct des valeurs CT (1-13) vers palette
7. **Python** = xarray avec engine='h5netcdf' pour compatibilité
