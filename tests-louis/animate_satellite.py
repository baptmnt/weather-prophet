"""
Script pour créer des animations GIF de tous les fichiers satellites NetCDF
"""
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import imageio
from matplotlib import colors as mcolors

# Chemins
DATA_DIR = Path("D:/Documents/Scolarité/5 - INSA Lyon/4TCA/S3/TIP/Projet/meteonet/data_samples/satellite")
OUTPUT_DIR = Path("./animations")
OUTPUT_DIR.mkdir(exist_ok=True)

# Classification des types de nuages
CLOUD_TYPES = {
    0: "No data", 1: "Cloud-free land", 2: "Cloud-free sea",
    3: "Snow over land", 4: "Sea ice", 5: "Very low clouds",
    6: "Low clouds", 7: "Mid-level clouds", 8: "High opaque clouds",
    9: "Very high opaque clouds", 10: "Fractional clouds",
    11: "High semitransparent thin", 12: "High semitransparent medium",
    13: "High semitransparent thick", 14: "High + low/medium",
    15: "High + snow/ice"
}

def create_animation(nc_file, fps=2, figsize=(10, 8)):
    """
    Crée une animation GIF à partir d'un fichier NetCDF
    
    Parameters:
    -----------
    nc_file : Path
        Chemin vers le fichier NetCDF
    fps : int
        Images par seconde dans le GIF (défaut: 2)
    figsize : tuple
        Taille de la figure (largeur, hauteur)
    """
    print(f"\n{'='*70}")
    print(f"Traitement de : {nc_file.name}")
    print(f"{'='*70}")
    
    # Charger les données
    data = xr.open_dataset(nc_file, engine='h5netcdf')
    var_name = list(data.data_vars)[0]
    var_data = data[var_name]
    
    # Informations
    n_times = len(data.time)
    print(f"Variable: {var_name}")
    print(f"Nombre d'images: {n_times}")
    print(f"Dimensions: {var_data.shape}")
    
    # Déterminer la palette de couleurs selon le type
    if var_name == 'CT':
        # Palette catégorielle pour Cloud Type
        n_cats = 16
        cmap = plt.cm.get_cmap('tab20', n_cats)
        norm = mcolors.BoundaryNorm(range(n_cats + 1), cmap.N)
        vmin, vmax = 0, 15
    else:
        # Palette continue pour autres canaux
        cmap = 'viridis'
        vmin = float(var_data.min())
        vmax = float(var_data.max())
        norm = None
    
    # Créer les frames
    frames = []
    temp_files = []
    
    for i in range(n_times):
        print(f"  Génération frame {i+1}/{n_times}...", end='\r')
        
        # Créer la figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Sélectionner le pas de temps
        frame_data = var_data.isel(time=i)
        time_str = str(data.time.values[i])[:19]  # Format: YYYY-MM-DD HH:MM:SS
        
        # Afficher l'image
        if norm:
            im = ax.imshow(frame_data, cmap=cmap, norm=norm, 
                          vmin=vmin, vmax=vmax, origin='upper',
                          extent=[float(data.lon.min()), float(data.lon.max()),
                                 float(data.lat.min()), float(data.lat.max())])
        else:
            im = ax.imshow(frame_data, cmap=cmap, 
                          vmin=vmin, vmax=vmax, origin='upper',
                          extent=[float(data.lon.min()), float(data.lon.max()),
                                 float(data.lat.min()), float(data.lat.max())])
        
        # Titre et labels
        ax.set_xlabel('Longitude (°E)', fontsize=11)
        ax.set_ylabel('Latitude (°N)', fontsize=11)
        ax.set_title(f'{var_name} - {time_str}\nFrame {i+1}/{n_times}', 
                    fontsize=13, fontweight='bold')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05)
        if var_name == 'CT':
            cbar.set_label('Cloud Type Category', fontsize=10)
        else:
            cbar.set_label(f'{var_name}', fontsize=10)
        
        # Sauvegarder temporairement
        temp_file = OUTPUT_DIR / f"temp_frame_{i:03d}.png"
        plt.savefig(temp_file, dpi=100, bbox_inches='tight')
        temp_files.append(temp_file)
        plt.close(fig)
        
        # Charger l'image pour le GIF
        frames.append(imageio.imread(temp_file))
    
    print(f"  Génération frame {n_times}/{n_times}... ✓")
    
    # Créer le GIF
    output_file = OUTPUT_DIR / f"{nc_file.stem}_animation.gif"
    print(f"\n  Création du GIF: {output_file.name}...")
    
    # Ajouter une pause à la fin
    frames.extend([frames[-1]] * fps)  # Pause de 1 seconde à la fin
    
    imageio.mimsave(output_file, frames, fps=fps, loop=0)
    
    # Nettoyer les fichiers temporaires
    print(f"  Nettoyage des fichiers temporaires...")
    for temp_file in temp_files:
        temp_file.unlink()
    
    file_size = output_file.stat().st_size / 1024  # KB
    print(f"  ✓ Animation créée: {output_file.name} ({file_size:.1f} KB)")
    print(f"  Durée: ~{n_times/fps:.1f}s à {fps} fps")
    
    return output_file

def create_comparison_animation(nc_files, fps=2):
    """
    Crée une animation avec plusieurs fichiers côte à côte
    """
    print(f"\n{'='*70}")
    print("CRÉATION D'UNE ANIMATION COMPARATIVE")
    print(f"{'='*70}")
    
    # Charger tous les fichiers
    datasets = []
    var_names = []
    for nc_file in nc_files:
        data = xr.open_dataset(nc_file, engine='h5netcdf')
        datasets.append(data)
        var_names.append(list(data.data_vars)[0])
    
    n_files = len(datasets)
    n_times = min(len(ds.time) for ds in datasets)
    
    print(f"Fichiers: {n_files}")
    print(f"Variables: {var_names}")
    print(f"Frames communes: {n_times}")
    
    frames = []
    temp_files = []
    
    # Créer layout selon nombre de fichiers
    if n_files <= 2:
        nrows, ncols = 1, n_files
        figsize = (12, 5)
    elif n_files <= 4:
        nrows, ncols = 2, 2
        figsize = (12, 10)
    else:
        nrows, ncols = 2, 3
        figsize = (15, 10)
    
    for i in range(n_times):
        print(f"  Frame {i+1}/{n_times}...", end='\r')
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        if n_files == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for j, (ds, var_name, ax) in enumerate(zip(datasets, var_names, axes)):
            frame_data = ds[var_name].isel(time=i)
            
            # Palette selon type
            if var_name == 'CT':
                cmap = plt.cm.get_cmap('tab20', 16)
                norm = mcolors.BoundaryNorm(range(17), cmap.N)
                vmin, vmax = 0, 15
            else:
                cmap = 'viridis'
                vmin = float(ds[var_name].min())
                vmax = float(ds[var_name].max())
                norm = None
            
            if norm:
                im = ax.imshow(frame_data, cmap=cmap, norm=norm, origin='upper')
            else:
                im = ax.imshow(frame_data, cmap=cmap, vmin=vmin, vmax=vmax, origin='upper')
            
            time_str = str(ds.time.values[i])[:16]
            ax.set_title(f'{var_name} - {time_str}', fontsize=10, fontweight='bold')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Masquer les axes inutilisés
        for j in range(n_files, len(axes)):
            axes[j].axis('off')
        
        plt.suptitle(f'Comparaison multi-canaux - Frame {i+1}/{n_times}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        temp_file = OUTPUT_DIR / f"comp_frame_{i:03d}.png"
        plt.savefig(temp_file, dpi=100, bbox_inches='tight')
        temp_files.append(temp_file)
        plt.close(fig)
        
        frames.append(imageio.imread(temp_file))
    
    print(f"  Frame {n_times}/{n_times}... ✓")
    
    # Créer le GIF
    output_file = OUTPUT_DIR / "comparison_animation.gif"
    print(f"\n  Création du GIF comparatif...")
    
    frames.extend([frames[-1]] * fps)
    imageio.mimsave(output_file, frames, fps=fps, loop=0)
    
    # Nettoyer
    for temp_file in temp_files:
        temp_file.unlink()
    
    file_size = output_file.stat().st_size / 1024
    print(f"  ✓ Animation créée: {output_file.name} ({file_size:.1f} KB)")
    
    return output_file

def main():
    print("="*70)
    print("GÉNÉRATEUR D'ANIMATIONS SATELLITE")
    print("="*70)
    
    # Lister tous les fichiers .nc
    nc_files = sorted(DATA_DIR.glob("*.nc"))
    print(f"\nFichiers trouvés: {len(nc_files)}")
    for f in nc_files:
        print(f"  - {f.name}")
    
    # Créer une animation pour chaque fichier
    print(f"\n{'='*70}")
    print("PHASE 1: ANIMATIONS INDIVIDUELLES")
    print(f"{'='*70}")
    
    output_files = []
    for nc_file in nc_files:
        try:
            output = create_animation(nc_file, fps=2)
            output_files.append(output)
        except Exception as e:
            print(f"  ✗ Erreur: {e}")
    
    # Créer une animation comparative avec les fichiers SE (même zone)
    print(f"\n{'='*70}")
    print("PHASE 2: ANIMATION COMPARATIVE (fichiers SE)")
    print(f"{'='*70}")
    
    se_files = [f for f in nc_files if '_SE_' in f.name]
    if len(se_files) >= 2:
        try:
            create_comparison_animation(se_files[:4], fps=2)  # Max 4 pour lisibilité
        except Exception as e:
            print(f"  ✗ Erreur: {e}")
    
    # Résumé
    print(f"\n{'='*70}")
    print("RÉSUMÉ")
    print(f"{'='*70}")
    print(f"\nAnimations créées: {len(output_files)}")
    print(f"Dossier de sortie: {OUTPUT_DIR.absolute()}")
    print("\nFichiers générés:")
    for output in OUTPUT_DIR.glob("*.gif"):
        size = output.stat().st_size / 1024
        print(f"  - {output.name} ({size:.1f} KB)")
    
    print(f"\n{'='*70}")
    print("✓ TERMINÉ!")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
