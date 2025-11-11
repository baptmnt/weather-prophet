# Fais un code permettant de récupérer les données d'une seule station météo
# et d l'enregistrer dans un fichier CSV

import pandas as pd
import argparse

def extract_single_station_data(station_id: str, input_filepath: str, output_filepath: str):
    """
    Extrait les données d'une seule station météo et les enregistre dans un fichier CSV.

    Args:
        station_id (str): L'ID de la station météo à extraire.
        input_filepath (str): Le chemin vers le fichier CSV contenant les données de toutes les stations.
        output_filepath (str): Le chemin vers le fichier CSV où enregistrer les données extraites.
    """
    # Lire le fichier CSV contenant les données de toutes les stations
    df = pd.read_csv(input_filepath)
    print("Données chargées depuis", input_filepath)

    # Filtrer les données pour ne garder que celles de la station spécifiée
    station_data = df[df['number_sta'] == int(station_id)]

    print(f"Données filtrées pour la station {station_id}, nombre de lignes: {len(station_data)}")  

    # Enregistrer les données filtrées dans un nouveau fichier CSV
    station_data.to_csv(output_filepath, index=False)

    print(f"Données de la station {station_id} extraites et enregistrées dans {output_filepath}")

if __name__ == "__main__":
    #Inline args
    parser = argparse.ArgumentParser(description="Extraire les données d'une seule station météo.")
    parser.add_argument("--station-id", type=str, help="L'ID de la station météo à extraire.")
    parser.add_argument("--input-filepath", type=str, help="Le chemin vers le fichier CSV contenant les données de toutes les stations.")
    parser.add_argument("--output-filepath", type=str, help="Le chemin vers le fichier CSV où enregistrer les données extraites.")
    args = parser.parse_args()

    extract_single_station_data(args.station_id, args.input_filepath, args.output_filepath)