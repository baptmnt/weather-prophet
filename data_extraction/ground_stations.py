# Code permettant d'extraire les données des stations au sol, avec plusieurs paramètres possibles
# Le but est de créer un dataset personnalisé en fonction des besoins de l'utilisateur. 
# Le dataset final associe une liste de relevés de stations spécifiques, à des dates relatives spécifiques (h-1, j-1, j-7, etc.) à des valeurs cibles
# sur la stations choisie (température, humidité, etc.)

# 1) Choix d'une station cible (ex: 69029001 pour la station de Bron)
# 2) Soit :
#    a) Choisir d'utiliser les données des stations proches (ex: dans un rayon de x km)
#    b) Choisir d'utiliser uniquement les données de la station cible
#    c) Choisir d'utiliser les données de plusieurs stations spécifiques (ex: une liste de numéros de stations)
# 3) Choix des paramètres à extraire (ex: humidité, température, etc.)
# 4) Choix de quelles dates extraires pour créer chaque data
# 5) Choix du fichier de sortie

# Le tout est paramétrable avec la commande ci-dessous
# python3 data_extraction/ground_stations.py --zone SE --years 2016 2017 2018 --station_cible 69029001 --station_filter 50 --extract_params hu t --extract_dates h-12 j-1 j-7 --output_path ../dataset/extract/SE/ground_stations


import pandas as pd
import numpy as np
import dask.dataframe as dd
from geopy.distance import geodesic
import matplotlib.pyplot as plt
from datetime import timedelta
import argparse
import os
from datetime import datetime

# Exemple de paramètres, ne pas prendre en compte ces valeurs, elles seront écrasées par les arguments de la ligne de commande
# ZONE = 'SE'  # Southeast zone
# YEARS = [2016, 2017, 2018]  # Years to process
# STATION_CIBLE = 69029001  # Station number for Bron

# STATION_FILTER = [69029001, 74001001, 74101001]  # Example list of specific stations
# STATION_FILTER = None  # Set to None to use only the target station
# STATION_FILTER = 50  # Radius in km for nearby stations

# EXTRACT_PARAMS = ['number_sta', 'dd', 'ff', 'precip', 'hu', 'td', 't', 'psl']  # Parameters to extract: humidity and temperature

# EXTRACT_DATES = ['h-12', 'j-1', 'j-7']  # Dates to extract relative to target date

# OUTPUT_PATH = "../dataset/extract/SE/ground_stations"  # Output path for extracted data

def get_timedelta_from_str(delta_str):
    """Convert a string representation of a time delta to a timedelta object."""
    if delta_str.startswith('h'):
        hours = int(delta_str[1:])
        return timedelta(hours=hours)
    elif delta_str.startswith('j'):
        days = int(delta_str[1:])
        return timedelta(days=days)
    else:
        raise ValueError("Invalid time delta format. Use 'h' for hours and 'j' for days.")

def get_time_deltas(delta_list):
    """Convert a list of string representations of time deltas to a list of timedelta objects."""
    return {delta: get_timedelta_from_str(delta) for delta in delta_list}

def fast_filter_by_radius(df, cible_coords, radius_km):
    """Vectorised haversine filter — accepts cible_coords=(lat, lon) in degrees."""
    if df is None or df.empty:
        return df
    R = 6371.0  # Earth radius in km
    lat1, lon1 = np.radians(tuple(cible_coords))
    # ensure numeric arrays
    lat2 = np.radians(df["lat"].astype(float).to_numpy())
    lon2 = np.radians(df["lon"].astype(float).to_numpy())
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    # numerical safety for asin(sqrt(a)) with clipping
    c = 2.0 * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0))
    distances = R * c
    mask = distances <= float(radius_km)
    return df.loc[mask]

def determine_allowed_stations(file_paths, station_cible, station_filter):
    """
    Build a list of allowed station numbers:
    - if station_filter is None -> only station_cible
    - if station_filter is int -> stations within radius km of station_cible (using pandas)
    - if station_filter is list -> that list
    file_paths: iterable of csv paths to scan for station metadata (number_sta, lat, lon)
    """
    # if explicit list given
    if isinstance(station_filter, list):
        return list(map(int, station_filter))

    # if None -> only target station
    if station_filter is None:
        return [int(station_cible)]

    # else station_filter is expected to be an int radius (km)
    radius_km = int(station_filter)

    # Regarder si on a pas déjà la donnée stockée localement dans temp_data/...
    cache_path = f"temp_data/{station_cible}_{radius_km}km.csv"
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path)
        return df.astype(int)['number_sta'].tolist()

    # collect station metadata (small) using pandas
    stations = []
    for fp in file_paths:
        if not os.path.exists(fp):
            continue
        try:
            df_meta = pd.read_csv(fp, usecols=['number_sta', 'lat', 'lon'])
            stations.append(df_meta)
            break
        except Exception:
            # ignore files that don't contain the expected columns
            continue
    if not stations:
        # fallback: just return the target station
        return [int(station_cible)]
    stations_df = pd.concat(stations, ignore_index=True).drop_duplicates(subset='number_sta')
    # find target coords
    target_row = stations_df[stations_df['number_sta'] == int(station_cible)]
    if target_row.empty:
        # fallback to only target id
        return [int(station_cible)]
    cible_coords = (float(target_row['lat'].iloc[0]), float(target_row['lon'].iloc[0]))
    nearby = fast_filter_by_radius(stations_df, cible_coords, radius_km)

    # save to cache for future runs
    os.makedirs("temp_data", exist_ok=True)
    nearby.to_csv(cache_path, index=False)

    return nearby['number_sta'].astype(int).tolist()

def load_filtered_data(file_path, allowed_station_ids, usecols=None):
    """Charge un CSV avec Dask et filtre par isin(allowed_station_ids)."""
    if not os.path.exists(file_path):
        # return empty dask dataframe with correct columns
        empty_pd = pd.DataFrame(columns=usecols)
        return dd.from_pandas(empty_pd, npartitions=1)

    # read with dask - parse 'date' column if present in usecols
    parse_dates = []
    if usecols and 'date' in usecols:
        parse_dates = ['date']

    # dd.read_csv returns a dask dataframe, not an iterator
    df = dd.read_csv(file_path, usecols=usecols, parse_dates=parse_dates, assume_missing=True)
    if not allowed_station_ids:
        empty_pd = pd.DataFrame(columns=usecols)
        return dd.from_pandas(empty_pd, npartitions=1)
    # filter using vectorised isin (works with dask)
    filtered = df[df['number_sta'].isin(allowed_station_ids)]
    return filtered

def hash_args_to_str(args):
    """Create a hash string from the extraction parameters for caching purposes."""
    import hashlib
    arg_str = "_".join(map(str, args))
    return hashlib.md5(arg_str.encode()).hexdigest()

def save_temp_data(data, output_path, filename, args_hash):
    """Save the extracted data to a Parquet file with a name based on the parameters hash."""
    os.makedirs(output_path, exist_ok=True)
    file_path = os.path.join(output_path, f"{filename}_{args_hash}.parquet")
    data.to_parquet(file_path, engine="pyarrow", compression="snappy")
    print(f"Extracted data saved to {file_path}")

def load_temp_data(output_path, filename, args_hash):
    """Load the extracted data from a Parquet file if it exists."""
    file_path = os.path.join(output_path, f"{filename}_{args_hash}.parquet")
    print(file_path)
    if os.path.exists(file_path):
        print(f"Loading cached data from {file_path}")
        return dd.read_parquet(file_path, engine="pyarrow")
    return None

# Définition des paramètres à partir des line-arguments
if __name__ == "__main__":
    ## Lecture des arguments de la ligne de commande
    parser = argparse.ArgumentParser(description="Extract ground station data.")
    parser.add_argument('--zone', type=str, required=True, help='Zone (e.g., SE)')
    parser.add_argument('--years', type=int, nargs='+', required=True, help='Years to process (e.g., 2016 2017 2018)')
    parser.add_argument('--station_cible', type=int, required=True, help='Target station number (e.g., 69029001)')
    parser.add_argument('--station_filter', type=str, required=False, help='Station filter: radius in km, list of station numbers, or None')
    parser.add_argument('--extract_params', type=str, nargs='+', required=True, help='Parameters to extract (e.g., hu t)')
    parser.add_argument('--extract_dates', type=str, nargs='+', required=True, help='Dates to extract (e.g., h-12 j-1 j-7)')
    parser.add_argument('--output_path', type=str, required=True, help='Output path for extracted data')
    args = parser.parse_args()

    ZONE = args.zone
    YEARS = args.years
    STATION_CIBLE = args.station_cible
    STATION_FILTER = args.station_filter
    if STATION_FILTER is not None:
        try:
            STATION_FILTER = int(STATION_FILTER)
        except ValueError:
            STATION_FILTER = [int(sta) for sta in STATION_FILTER.split(',')]
    EXTRACT_PARAMS = args.extract_params
    EXTRACT_DATES = get_time_deltas(args.extract_dates)
    OUTPUT_PATH = args.output_path

    hash_args = hash_args_to_str([ZONE, YEARS, STATION_CIBLE, STATION_FILTER, EXTRACT_PARAMS, args.extract_dates])


    print(f"Parameters:\n Zone: {ZONE}\n Years: {YEARS}\n Target Station: {STATION_CIBLE}\n Station Filter: {STATION_FILTER}\n Extract Params: {EXTRACT_PARAMS}\n Extract Dates: {args.extract_dates}\n Output Path: {OUTPUT_PATH}")

    cached_data = load_temp_data("temp_data", "1h_ground_stations",hash_args)
    if cached_data is not None:
        print(f"Data already extracted and cached. Loaded {int(cached_data.shape[0].compute())} records.")
        data = cached_data


    else:
        print("Starting data extraction...")
        t1 = datetime.now()

        # Check for cached data
        cached_data = load_temp_data("temp_data", "filtered_ground_stations",hash_args)
        if cached_data is not None:
            print("Using cached filtered data.")
            data = cached_data
            n_records = int(data.shape[0].compute())
        else:

            usecols = ['number_sta', 'lat', 'lon', 'date'] + EXTRACT_PARAMS

            # Build file paths list to determine allowed stations (small pandas ops)
            file_paths = [f"../dataset/{ZONE}/ground_stations/{ZONE}{year}.csv" for year in YEARS]
            allowed_station_ids = determine_allowed_stations(file_paths, STATION_CIBLE, STATION_FILTER)
            t2 = datetime.now()
            print(f"Allowed stations ({len(allowed_station_ids)}) in {(t2-t1).total_seconds()}s: {allowed_station_ids[:10]}{'...' if len(allowed_station_ids)>10 else ''}")

            dask_dfs = []
            for year in YEARS:
                file_path = f"../dataset/{ZONE}/ground_stations/{ZONE}{year}.csv"
                print(f"Processing {file_path} ...")
                df_filtered = load_filtered_data(file_path, allowed_station_ids, usecols=usecols)
                dask_dfs.append(df_filtered)

            # concat dask dataframes (skip if none)
            if dask_dfs:
                data = dd.concat(dask_dfs, interleave_partitions=True)
            else:
                data = dd.from_pandas(pd.DataFrame(columns=usecols), npartitions=1)

            # compute number of records (this triggers computation)
            try:
                n_records = int(data.shape[0].compute())
            except Exception:
                n_records = 0

            # Save to temp data for future runs
            save_temp_data(data, "temp_data", "filtered_ground_stations", hash_args)
        
            t3 = datetime.now()
            print(f"Data loaded and filtered in {(t3-t2).total_seconds()}s: {n_records} records")


        # Reduce data to one record per hour (instead of 4 at differents minutes) per station (keep first)
        data["date_hour"] = data["date"].dt.floor("1h")

        data = data.drop_duplicates(subset=['number_sta', 'date_hour'])
        data = data.drop(columns=['date_hour'])

        save_temp_data(data, "temp_data", "1h_ground_stations", hash_args)

        t4 = datetime.now()
        print(f"Data reduced to hourly records in {(t4-t3).total_seconds()}s : {int(data.shape[0].compute())} records")

    t4 = datetime.now()
    print("Starting creation of extracted dataset...")

    stations_autour = determine_allowed_stations(
        [f"../dataset/{ZONE}/ground_stations/{ZONE}{year}.csv" for year in YEARS],
        STATION_CIBLE,
        STATION_FILTER
    )

    df_target = data[data["number_sta"] == STATION_CIBLE].set_index("date")
    dataset = df_target.copy()
    dataset.index.name = "date"

    feature_dfs = []

    # Boucle sur deltas et stations autour
    for delta_name, delta_td in EXTRACT_DATES.items():
        print(f"Processing features for delta {delta_name}...")
        for sta in stations_autour:
            # Extraire relevés de la station
            df_sta = data[data["number_sta"] == sta][["date"] + EXTRACT_PARAMS].copy()

            # Décaler la date
            df_sta["date"] = df_sta["date"] + delta_td

            # Renommer colonnes
            df_sta = df_sta.rename(columns={param: f"{sta}_{param}_{delta_name}" for param in EXTRACT_PARAMS})

            # Index sur date
            df_sta = df_sta.set_index("date")

            feature_dfs.append(df_sta)
    print("Combining all features...")
    features_all = dd.concat(feature_dfs, axis=1)

    # Joindre avec la station cible
    dataset = dataset.join(features_all, how="left")

    # Compter nombre de colonnes
    print(f"Final dataset has {int(dataset.shape[1])} columns.")

    print("Saving extracted dataset...")
    dataset.compute().to_csv(os.path.join("temp_data", f"extracted_ground_stations_{hash_args}.csv"), index=False)

    save_temp_data(dataset, "temp_data", "extracted_ground_stations", hash_args)

    # Save also to csv
    t5 = datetime.now()
    print(f"Extracted dataset created and saved in {(t5-t4).total_seconds()}s : {int(dataset.shape[0].compute())} records")


