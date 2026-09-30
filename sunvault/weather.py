"""Site weather (F1): synthetic presets, hourly arrays, or CSV (NASA POWER / IMD / logger)."""
from dataclasses import dataclass
import numpy as np
from .model import DT

SITES = {  # mean C, daily swing C, peak solar on south glazing W/m2 (synthetic presets)
    "Leh - October":     dict(m=4.0,  a=9.0, g=700.0),
    "Leh - January":     dict(m=-8.0, a=8.0, g=600.0),
    "Tawang - January":  dict(m=2.0,  a=5.0, g=400.0),
    "Srinagar - January": dict(m=0.5, a=6.0, g=450.0),
    "Shimla - January":  dict(m=3.0,  a=6.0, g=500.0),
    "Jaisalmer - May":   dict(m=34.0, a=9.0, g=900.0),
    "Gorakhpur - June":  dict(m=33.0, a=5.0, g=800.0),
}
CLIMATE = {"Leh": "Cold-dry, high altitude", "Tawang": "Cold, cloudy", "Srinagar": "Cold, temperate",
           "Shimla": "Cold, sub-montane", "Jaisalmer": "Hot-dry desert", "Gorakhpur": "Hot, composite plains"}

@dataclass(eq=False)
class Weather:
    hour: np.ndarray   # hour of day at every model step
    Ta: np.ndarray     # outside temperature, C
    G: np.ndarray      # solar irradiance on the glazing, W/m2
    n_spin: int        # spin-up steps that are dropped from results
    Ta_h: np.ndarray = None   # original hourly series (for export)
    G_h: np.ndarray = None

def from_hourly(Ta_h, G_h, spin_days=2):
    Ta_h, G_h = np.asarray(Ta_h, float), np.asarray(G_h, float)
    assert len(Ta_h) == len(G_h) and len(Ta_h) % 24 == 0, "need whole days of hourly data"
    Ta_s = np.concatenate([np.tile(Ta_h[:24], spin_days), Ta_h])
    G_s = np.concatenate([np.tile(G_h[:24], spin_days), G_h])
    n = int(len(Ta_s) * 3600 / DT)
    t = np.arange(n) * DT / 3600
    th = np.arange(len(Ta_s))
    return Weather(hour=t % 24, Ta=np.interp(t, th, Ta_s), G=np.interp(t, th, G_s),
                   n_spin=int(spin_days * 86400 / DT), Ta_h=Ta_h, G_h=G_h)

def synthetic(site, days=1, cloud=0.0, dT=0.0):
    s = SITES[site]
    cloud = np.broadcast_to(np.asarray(cloud, float), (days,))
    h = np.arange(24 * days)
    hod = h % 24
    Ta = s["m"] + dT + s["a"] * np.cos(2 * np.pi * (hod - 15) / 24)
    G = s["g"] * (1 - 0.8 * cloud[h // 24]) * np.where((hod >= 7) & (hod <= 17), np.sin(np.pi * (hod - 7) / 10), 0.0)
    return from_hourly(Ta, np.clip(G, 0, None))

def from_csv(path_or_file):
    """Hourly CSV with a temperature column (temp_c or T2M) and a solar column (ghi_wm2 or ALLSKY_SFC_SW_DWN)."""
    import pandas as pd
    df = pd.read_csv(path_or_file)
    tc = next(c for c in ("temp_c", "T2M") if c in df.columns)
    gc = next(c for c in ("ghi_wm2", "ALLSKY_SFC_SW_DWN") if c in df.columns)
    n = len(df) // 24 * 24
    return from_hourly(df[tc].values[:n], df[gc].values[:n])
