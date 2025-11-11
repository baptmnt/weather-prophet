import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    """
    Modèle CNN pour la prédiction [humidité, température] à partir d'images satellite météo.
    - Entrée : image RGB (3 canaux), taille 171x261
    - Sortie : vecteur (2 valeurs)
    """

    def __init__(self, output_dim=2):
        super().__init__()

        # --- Bloc convolutionnel ---
        # On empile plusieurs couches convolutionnelles + normalisation + activation
        # Ces blocs apprennent des motifs spatiaux à différentes échelles (textures, nuages, zones thermiques, etc.)

        self.features = nn.Sequential(
            # Première couche : extrait des motifs simples (bords, transitions, nuages clairs/sombres)
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  
            nn.BatchNorm2d(32),     # Stabilise la distribution des activations
            nn.ReLU(inplace=True),  # Activation non-linéaire pour extraire des caractéristiques complexes
            nn.MaxPool2d(2, 2),     # Réduit la taille spatiale → 171x261 devient ~85x130

            # Deuxième couche : apprend des motifs intermédiaires (zones météo régionales)
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),     # ~42x65

            # Troisième couche : motifs globaux (grandes masses nuageuses, fronts)
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),     # ~21x32

            # Quatrième couche : abstrait encore plus, peu de paramètres spatiaux
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),     # ~10x16
        )

        # --- Global Average Pooling ---
        # Permet d’obtenir un vecteur invariant à la taille d’image d’entrée
        # (cela évite de dépendre d’un "magic number" comme 64*8*8)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.flatten = nn.Flatten() pourrait aussi être utilisé si l'image d'entrée est carrée

        # --- Couches fully connected ---
        # C’est ici que le réseau apprend à lier les features visuelles à la température/humidité
        self.regressor = nn.Sequential(
            nn.Linear(256, 128),  # 256 vient du nombre de filtres de la dernière couche conv
            # si on utilise flatten : nn.Linear(256*x*y, 128)
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),      # Régularisation pour éviter le surapprentissage : on "éteint" aléatoirement 30% des neurones à chaque itération
            nn.Linear(128, output_dim)
        )

        # --- Initialisation des poids (meilleure convergence) ---
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # --- Étape 1 : extraire les features visuelles : Pooling indépendant de la taille d'entrée ---
        x = self.features(x)              # [B, 256, ~10, ~16]

        # --- Étape 2 : global average pooling pour compacter chaque carte de feature ---
        x = self.global_pool(x)           # [B, 256, 1, 1]

        # --- Étape 3 : flatten pour entrer dans le réseau fully connected : passer de [B, C, 1, 1] à [B, C] ---
        x = x.view(x.size(0), -1)         # [B, 256] 

        # --- Étape 4 : régresseur linéaire pour prédire (humid, temp) ---
        x = self.regressor(x)             # [B, 2]

        return x
