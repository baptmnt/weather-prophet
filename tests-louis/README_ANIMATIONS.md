# 🎬 Animations Satellites Générées

## 📁 Fichiers créés

Toutes les animations sont dans le dossier : `animations/`

### Animations individuelles (24 heures, 1er janvier 2016)

1. **`IR039_SE_2016_animation.gif`** (1.2 MB)
   - **Canal** : Infrarouge 3.9 µm
   - **Mesure** : Température de brillance (°C)
   - **Durée** : ~12s à 2 fps (24 images)
   - **Zone** : Sud-Est France
   - **Usage** : Détection de nuages, température des sommets nuageux

2. **`IR108_SE_2016_animation.gif`** (1.5 MB)
   - **Canal** : Infrarouge 10.8 µm
   - **Mesure** : Température de brillance (°C)
   - **Durée** : ~12s à 2 fps (24 images)
   - **Zone** : Sud-Est France
   - **Usage** : Canal principal pour température, fonctionne jour/nuit

3. **`VIS06_SE_2016_animation.gif`** (535 KB)
   - **Canal** : Visible 0.635 µm
   - **Mesure** : Radiance/réflectance (%)
   - **Durée** : ~12s à 2 fps (24 images)
   - **Zone** : Sud-Est France
   - **Usage** : Albédo des nuages, épaisseur nuageuse (jour uniquement)
   - **⚠️ Note** : Noir la nuit (valeurs manquantes)

4. **`WV062_SE_2016_animation.gif`** (1.1 MB)
   - **Canal** : Vapeur d'eau 6.25 µm
   - **Mesure** : Température de brillance (°C)
   - **Durée** : ~12s à 2 fps (24 images)
   - **Zone** : Sud-Est France
   - **Usage** : Contenu en humidité atmosphérique, altitude des masses humides

### Animation comparative

1. **`comparison_animation.gif`** (taille variable)
   - **Contenu** : Les 4 canaux côte à côte
   - **Layout** : Grille 2×2
   - **Durée** : ~12s à 2 fps
   - **Usage** : Comparaison directe de tous les canaux

---

## 🔍 Comment interpréter chaque animation

### IR039 & IR108 (Infrarouges)

**Code couleur** :

- **Violet/Bleu foncé** : Très froid (~-60°C à -40°C) → Nuages très hauts (convection profonde, orages)
- **Bleu/Cyan** : Froid (~-40°C à -20°C) → Nuages hauts (cirrus, cirrostratus)
- **Vert/Jaune** : Modéré (~-20°C à 0°C) → Nuages moyens/bas
- **Orange/Rouge** : Chaud (~0°C à 20°C) → Sol, mer, zones dégagées

**Différence IR039 vs IR108** :

- **IR039** (3.9 µm) : Plus sensible aux petites gouttes, utile de nuit
- **IR108** (10.8 µm) : Canal standard, meilleure température globale

**Ce qu'on observe** :

- Déplacement des systèmes nuageux
- Développement/dissipation de convection
- Fronts chauds/froids
- Évolution jour/nuit de la température de surface

---

### VIS06 (Visible)

**Code couleur** :

- **Noir** : Nuit ou ciel dégagé (0% réflectance)
- **Gris foncé** : Faible réflectance (mer, sol sombre, peu de nuages)
- **Gris clair** : Nuages fins ou fragmentés
- **Blanc** : Nuages épais (forte réflectance, 80-100%)

**Ce qu'on observe** :

- **Lever/coucher du soleil** : Gradient progressif d'éclairement
- **Épaisseur nuageuse** : Plus blanc = plus épais
- **Limite jour/nuit** : Zone noire vs zone éclairée
- **Évolution diurne** : Développement de cumulus l'après-midi

**⚠️ Limites** :

- Inutilisable la nuit
- Dépendant de l'angle solaire
- Difficile à interpréter au crépuscule

---

### WV062 (Vapeur d'eau)

**Code couleur** (généralement noir & blanc inversé) :

- **Blanc/Clair** : Atmosphère humide en altitude (beaucoup de vapeur d'eau)
- **Gris** : Humidité moyenne
- **Noir/Foncé** : Atmosphère sèche (air descendant, dorsales anticycloniques)

**Ce qu'on observe** :

- **Masses d'air** : Zones humides vs sèches
- **Jets streams** : Bandes sombres = air sec descendant
- **Flux d'humidité** : Transport vers zones pré-convectives
- **Frontogénèse** : Contraste sec/humide aux fronts

**Altitude** : Sensible à la couche 600-350 hPa (~4-8 km)

---

## 🎯 Analyse comparative (GIF multi-canaux)

En regardant les 4 canaux simultanément, on peut :

### 1. Identifier les types de nuages

| Type de nuage | VIS06 | IR108 | WV062 |
|---------------|-------|-------|-------|
| **Cirrus fins** | Gris clair | Froid moyen | Variable |
| **Cirrostratus épais** | Blanc | Très froid | Humide |
| **Cumulus** | Blanc compact | Modéré | Humide localisé |
| **Cumulonimbus** | Très blanc | Très froid | Très humide |
| **Stratus/Brouillard** | Gris uniforme | Chaud | Humide bas |

### 2. Distinguer jour/nuit

- **VIS06** : Noir la nuit → bascule sur IR108
- **IR108 + IR039** : Fonctionnent 24h/24
- **WV062** : Fonctionne toujours mais interprétation constante

### 3. Détecter les précipitations

**Signature typique** :

- VIS06 : Blanc intense (nuages épais)
- IR108 : Froid (<-30°C, sommet haut)
- WV062 : Très humide
- IR039 : Contraste texture (petites gouttes)

### 4. Suivre les fronts

**Front chaud** :

- Progression lente de nébulosité étendue
- WV062 montre l'advection d'air humide
- IR108 : Réchauffement progressif

**Front froid** :

- Ligne de convection nette
- Contraste fort en WV062
- Développement rapide en VIS06 (si jour)

---

## 💻 Script utilisé

Le script `animate_satellite.py` :

- Charge chaque fichier NetCDF avec `xarray`
- Génère une image par pas de temps
- Assemble les images en GIF avec `imageio`
- Palette automatique adaptée au type de canal
- Nettoyage automatique des fichiers temporaires

**Pour relancer** :

```bash
python animate_satellite.py
```

---

## 🚀 Prochaines étapes possibles

### Améliorations des animations

1. **Ajouter un overlay géographique**
   - Côtes, frontières, villes
   - Utiliser `cartopy` ou `basemap`

2. **Palette de couleurs personnalisée**
   - Palettes météo standards (MSG, SEVIRI)
   - Échelles de températures normalisées

3. **Annotations dynamiques**
   - Heure locale
   - Statistiques (min/max/moyenne)
   - Détection automatique de features

4. **Résolution variable**
   - GIF haute résolution pour impressions
   - GIF léger pour web
   - Format vidéo MP4

5. **Animations plus longues**
   - Charger plusieurs fichiers consécutifs
   - Animation sur plusieurs jours
   - Boucle saisonnière

### Analyses avancées

1. **Détection automatique d'événements**
   - Orages (IR < -50°C + développement rapide)
   - Fronts (gradients en WV062)
   - Brouillard (VIS06 gris uniforme + IR108 chaud)

2. **Tracking de systèmes**
   - Suivi de cellules convectives
   - Vitesse et direction du déplacement
   - Prévision à très court terme

3. **Fusion de canaux**
   - RGB composite (comme produits MSG)
   - Fausse couleur pour améliorer contraste
   - Produits dérivés (e.g., BTD = IR108-IR039)

4. **Validation de modèles**
   - Superposition prévisions AROME
   - Calcul d'erreurs
   - Ajustement de biais

---

## 📊 Statistiques des animations

| Fichier | Taille | Frames | Durée | Zone | Période |
|---------|--------|--------|-------|------|---------|
| IR039 | 1.2 MB | 24 | 12s | SE | 2016-01-01 |
| IR108 | 1.5 MB | 24 | 12s | SE | 2016-01-01 |
| VIS06 | 535 KB | 24 | 12s | SE | 2016-01-01 |
| WV062 | 1.1 MB | 24 | 12s | SE | 2016-01-01 |
| Comparaison | ~5 MB | 24 | 12s | SE | 2016-01-01 |

**Total** : ~9.3 MB pour 5 animations

---

## 🎓 Pour aller plus loin

### Documentation

- [MSG Interpretation Guide](http://www.eumetrain.org/msg_interpretation/) (EUMETSAT)
- [Satellite Meteorology Course](http://cimss.ssec.wisc.edu/)
- [MeteoNet Documentation](https://meteofrance.github.io/meteonet/)

### Outils complémentaires

- **Satpy** : Manipulation avancée de données satellite
- **PyTroll** : Suite complète pour traitement satellite
- **MetPy** : Calculs météorologiques
- **PyART** : Analyse radar (complémentaire)

### Projets possibles

1. **Nowcasting** : Prédiction 0-3h à partir des animations
2. **Classification** : ML pour identifier types de temps
3. **Composite RGB** : Créer des produits MSG-like
4. **Dashboard temps réel** : Interface web interactive
5. **Alerte automatique** : Détection d'orages, brouillard, etc.
