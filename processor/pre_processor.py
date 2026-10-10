from CONSTANTS import WGS84_OMEGA_E, C, ELEVATION_MASK, IONOFREE_PAIRS, FREQUENCIES, WAVELENGTHS
import numpy as np
import pandas as pd
from utils import ecef_to_geodetic, glonass_frequency
from hatchfilter import HatchFilter

def _resolve_code(obs: pd.DataFrame, code: str) -> tuple[str|None, float]:
    """
    Resolves a band (e.g. 'C5') to an observable of the epoch.
    Returns the first non-NaN column, in header order, that shares the band
    prefix (first two characters), with no criterion on the tracking
    attribute. Returns (code_used, value) or (None, NaN).
    """

    # Search by band prefix (first two characters)
    prefix = code[:2]
    for col in obs.index:
        if col.startswith(prefix) and col != code:
            val = obs.get(col)
            if val is not None and not pd.isna(val):
                return col, val

    return None, np.nan

def get_ionofree_pseudorange_smoothed(obs: pd.DataFrame, sv: str, hatch_filter: HatchFilter, freqnum: np.float64 = np.nan):
    """
    Ionosphere-free pseudorange with Hatch smoothing.
    """

    system = sv[0]

    for code1, code2 in IONOFREE_PAIRS[system]:
        code1_used, P1 = _resolve_code(obs, code1)
        code2_used, P2 = _resolve_code(obs, code2)

        if code1_used is None or code2_used is None:
            continue

        # Phases matching the code actually found
        phase1 = "L" + code1_used[1:]
        phase2 = "L" + code2_used[1:]

        L1 = obs.get(phase1) if phase1 else None
        L2 = obs.get(phase2) if phase2 else None
        
        # Smooth if phase is available
        if L1 is not None and not pd.isna(L1):
            wl1 = WAVELENGTHS[system][code1] if np.isnan(freqnum) else C/glonass_frequency(int(code1[1]), freqnum)
            P1 = hatch_filter.update(sv, code1, P1, L1, wl1)
        
        if L2 is not None and not pd.isna(L2):
            wl2 = WAVELENGTHS[system][code2] if np.isnan(freqnum) else C/glonass_frequency(int(code2[1]), freqnum)
            P2 = hatch_filter.update(sv, code2, P2, L2, wl2)
        
        # Ionosphere-free combination of the smoothed codes
        f1 = FREQUENCIES[system][code1] if np.isnan(freqnum) else glonass_frequency(int(code1[1]), freqnum)
        f2 = FREQUENCIES[system][code2] if np.isnan(freqnum) else glonass_frequency(int(code2[1]), freqnum)
        
        return ionofree_combination(P1, P2, f1, f2)
    
    return None


def ionofree_combination(p1: np.float64, p2: np.float64, f1: np.float64, f2: np.float64) -> np.float64:
    f1_sq = f1 ** 2
    f2_sq = f2 ** 2
    return (f1_sq * p1 - f2_sq * p2) / (f1_sq - f2_sq)
    

def get_ionofree_pseudorange(obs_epoch: pd.DataFrame, sv: str, code=True):
    
    system = sv[0]

    for s1, s2 in IONOFREE_PAIRS[system]:

        if pd.notna(obs_epoch[s1]) and pd.notna(obs_epoch[s2]):
            p1 = obs_epoch[s1]
            f1 = FREQUENCIES[system][s1]
            p2 = obs_epoch[s2]
            f2 = FREQUENCIES[system][s2]
            return ionofree_combination(p1, p2, f1, f2)

    return None

def calculate_elevation_azimuth(station_ecef, satellite_ecef):
    """
    Elevation and azimuth of the satellite as seen from the station.
    
    Parameters:
    -----------
    station_ecef : array-like [x, y, z]
        Station position in ECEF (metres)
    satellite_ecef : array-like [x, y, z]
        Satellite position in ECEF (metres)
    
    Returns:
    --------
    tuple : (elevation rad, azimuth rad)
    """
    
    x, y, z = station_ecef
    xs, ys, zs = satellite_ecef
    
    # Station -> satellite vector
    dx = xs - x
    dy = ys - y
    dz = zs - z
    
    # Geodetic coordinates of the station
    lat, lon, _ = ecef_to_geodetic(x, y, z)
    
    # ECEF -> ENU rotation
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)
    
    # ENU components (East, North, Up)
    E = -sin_lon * dx + cos_lon * dy
    N = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    U = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz
    
    # Horizontal distance
    horizontal_dist = np.sqrt(E**2 + N**2)
    
    # Elevation
    elevation = np.arctan2(U, horizontal_dist)
    
    # Azimuth (from North, positive towards East)
    azimuth = np.arctan2(E, N)
    if azimuth < 0:
        azimuth += 2 * np.pi
    
    return elevation, azimuth


def check_elevation_mask(station_ecef: list[np.float64], satellite_ecef: list[np.float64], mask=ELEVATION_MASK):
    """
    Checks whether the satellite is above the elevation mask.
    
    Returns:
    --------
    tuple : (valid bool, elevation rad, azimuth rad)
    """
    
    elevation, azimuth = calculate_elevation_azimuth(station_ecef, satellite_ecef)    
    return elevation >= mask, elevation, azimuth


# Tropospheric correction (Saastamoinen)
def standard_atmosphere(height_m: np.float64) -> tuple[np.float64, np.float64, np.float64]:
    
    P0 = 1013.25  # hPa at sea level
    T0 = 288.15   # K (15°C)
    L = 0.0065    # Temperature lapse rate K/m
    g = 9.80665   # m/s²
    M = 0.0289644 # kg/mol
    R = 8.31447   # J/(mol·K)
    
    T = T0 - L * height_m
    P = P0 * (T / T0) ** (g * M / (R * L))
    
    # Standard relative humidity (decreasing with height)
    H = 50.0 * np.exp(-height_m / 2000)
    
    return P, T, H


def saastamoinen_tropospheric_delay(elevation_rad, lat_rad, height_m):
    """
    Tropospheric delay with the Saastamoinen model and a standard atmosphere.
    
    Parameters:
    -----------
    elevation_rad : float
        Satellite elevation angle (radians)
    lat_rad : float
        Station latitude (radians)
    height_m : float
        Station ellipsoidal height (metres)
    
    Returns:
    --------
    float : Tropospheric delay in metres
    """
    
    
    pressure_hPa, temperature_K, humidity_percent = standard_atmosphere(height_m)
        
    # Avoid very low elevations
    #if elevation_rad < np.radians(5):
        #elevation_rad = np.radians(5)
    
    # Partial water vapour pressure (Magnus-Tetens formula)
    T_celsius = temperature_K - 273.15
    e_sat = 6.112 * np.exp((17.67 * T_celsius) / (T_celsius + 243.5))
    e = (humidity_percent / 100.0) * e_sat
    
    # Latitude and height correction factor
    f_lat = 1 - 0.00266 * np.cos(2 * lat_rad) - 0.00028 * (height_m / 1000)
    
    # Zenith hydrostatic delay (ZHD)
    ZHD = 0.0022768 * pressure_hPa / f_lat
    
    # Zenith wet delay (ZWD)
    ZWD = 0.0022768 * (1255 / temperature_K + 0.05) * e / f_lat
    
    # Total zenith delay
    ZTD = ZHD + ZWD
    
    # Simple mapping function (1/sin(elevation))
    mapping = 1.0 / np.sin(elevation_rad)
    
    # Slant delay
    tropo_delay = ZTD * mapping
    
    return tropo_delay


# Sagnac correction
def apply_sagnac_correction(satellite_ecef: list, station_ecef: list):
    xs, ys, zs = satellite_ecef
    xr, yr, zr = station_ecef
    
    # Approximate range
    dist = np.sqrt((xs - xr)**2 + (ys - yr)**2 + (zs - zr)**2)
    
    # Signal travel time
    tau = dist / C
    
    # Earth rotation angle during the travel time
    theta = WGS84_OMEGA_E * tau
    
    # Rotate the satellite position (small rotation)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    xs_corr = cos_theta * xs + sin_theta * ys
    ys_corr = -sin_theta * xs + cos_theta * ys
    zs_corr = zs
    
    return np.array([xs_corr, ys_corr, zs_corr])
