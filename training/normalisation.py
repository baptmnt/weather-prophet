#dd 	Wind direction 	degrees (°)
# ff 	Wind speed 	m.s-1
# precip 	Precipitation during the reporting period 	kg.m2
# hu 	Humidity 	percentage (%)
# td 	Dew point 	Kelvin (K)
# t 	Temperature 	Kelvin (K)
# psl 	Pressure reduced to sea level 	Pascal (Pa)

#Propose moi 7 fonctions de normalisation pour dd, ff, etc. 

def normalize_dd(dd):
    """Normalise la direction du vent (dd) entre 0 et 1."""
    return dd / 360.0

def denormalize_dd(normalized_dd):
    """Dénormalise la direction du vent (dd) de 0-1 à 0-360 degrés."""
    return normalized_dd * 360.0

def normalize_ff(ff, max_ff=60.0):
    """Normalise la vitesse du vent (ff) entre 0 et 1 en supposant une vitesse maximale de 60 m/s."""
    return ff / max_ff


def denormalize_ff(normalized_ff, max_ff=60.0):
    """Dénormalise la vitesse du vent (ff) de 0-1 à 0-max_ff m/s."""
    return normalized_ff * max_ff

def normalize_precip(precip, max_precip=500.0):
    """Normalise les précipitations (precip) entre 0 et 1 en supposant une valeur maximale de 500 mm."""
    return precip / max_precip

def denormalize_precip(normalized_precip, max_precip=500.0):
    """Dénormalise les précipitations (precip) de 0-1 à 0-max_precip mm."""
    return normalized_precip * max_precip

def normalize_hu(hu):
    """Normalise l'humidité (hu) entre 0 et 1."""
    return hu / 100.0

def denormalize_hu(normalized_hu):
    """Dénormalise l'humidité (hu) de 0-1 à 0-100%."""
    return normalized_hu * 100.0

def normalize_td(td, min_td=180.0, max_td=330.0):
    """Normalise le point de rosée (td) entre 0 et 1 en supposant une plage de température de 180K à 330K."""
    return (td - min_td) / (max_td - min_td)

def denormalize_td(normalized_td, min_td=180.0, max_td=330.0):
    """Dénormalise le point de rosée (td) de 0-1 à la plage min_td-max_td."""
    return normalized_td * (max_td - min_td) + min_td - 273.15  # Convertir de Kelvin à Celsius

def denormalize_residual_td(normalized_residual_td, min_td=180.0, max_td=330.0):
    """Dénormalise un résidu de point de rosée (td) de 0-1 à la plage min_td-max_td."""
    return normalized_residual_td * (max_td - min_td)  # Résidu en Kelvin

def normalize_t(t, min_t=180.0, max_t=330.0):
    """Normalise la température (t) entre 0 et 1 en supposant une plage de température de 180K à 330K."""
    return (t - min_t) / (max_t - min_t)

def denormalize_t(normalized_t, min_t=180.0, max_t=330.0):
    """Dénormalise la température (t) de 0-1 à la plage min_t-max_t."""
    return normalized_t * (max_t - min_t) + min_t - 273.15  # Convertir de Kelvin à Celsius

def denormalize_residual_t(normalized_residual_t, min_t=180.0, max_t=330.0):
    """Dénormalise un résidu de température (t) de 0-1 à la plage min_t-max_t."""
    return normalized_residual_t * (max_t - min_t)  # Résidu en Kelvin

def normalize_psl(psl, min_psl=87000.0, max_psl=108000.0):
    """Normalise la pression réduite au niveau de la mer (psl) entre 0 et 1 en supposant une plage de 87000 Pa à 108000 Pa."""
    return (psl - min_psl) / (max_psl - min_psl)

def denormalize_psl(normalized_psl, min_psl=87000.0, max_psl=108000.0):
    """Dénormalise la pression réduite au niveau de la mer (psl) de 0-1 à la plage min_psl-max_psl."""
    return (normalized_psl * (max_psl - min_psl) + min_psl)/1000.0  # Convertir de Pa à hPa

def denormalize_residual_psl(normalized_residual_psl, min_psl=87000.0, max_psl=108000.0):
    """Dénormalise un résidu de pression réduite au niveau de la mer (psl) de 0-1 à la plage min_psl-max_psl."""
    return normalized_residual_psl * (max_psl - min_psl) / 100.0 # Résidu en hPa


normalization_functions = {
    'dd': normalize_dd,
    'ff': normalize_ff,
    'precip': normalize_precip,
    'hu': normalize_hu,
    'td': normalize_td,
    't': normalize_t,
    'psl': normalize_psl
}
def normalize_variable(var_name, data):
    """Normalise une variable météorologique donnée en utilisant la fonction appropriée."""
    if var_name in normalization_functions:
        return normalization_functions[var_name](data)
    else:
        raise ValueError(f"Fonction de normalisation non définie pour la variable: {var_name}")
    
denormalization_functions = {
    'dd': denormalize_dd,
    'ff': denormalize_ff,
    'precip': denormalize_precip,
    'hu': denormalize_hu,
    'td': denormalize_td,
    't': denormalize_t,
    'psl': denormalize_psl
}
def denormalize_variable(var_name, normalized_data):
    """Dénormalise une variable météorologique donnée en utilisant la fonction appropriée."""
    if var_name in denormalization_functions:
        return denormalization_functions[var_name](normalized_data)
    else:
        raise ValueError(f"Fonction de dénormalisation non définie pour la variable: {var_name}")

denormalized_residual_functions = {
    'dd': denormalize_dd,
    'ff': denormalize_ff,
    'precip': denormalize_precip,
    'hu': denormalize_hu,
    'td': denormalize_residual_td,
    't': denormalize_residual_t,
    'psl': denormalize_residual_psl
}

def denormalize_residual_variable(var_name, normalized_residual):
    """Dénormalise un résidu d'une variable météorologique donnée en utilisant la fonction appropriée."""
    if var_name in denormalized_residual_functions:
        return denormalized_residual_functions[var_name](normalized_residual)
    else:
        raise ValueError(f"Fonction de dénormalisation des résidus non définie pour la variable: {var_name}")
