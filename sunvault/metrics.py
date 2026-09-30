"""Night Comfort Ratio and supporting metrics (F5, F6)."""
import numpy as np
from .model import xp, geom, make_sim, DT, Shelter, BASELINE
from .materials import MATERIALS, PRICE_INS, PRICE_PCM, PRICE_WIN

BAND = (18.0, 26.0)
KWH_PER_KG = 7.5            # diesel/kerosene, ~75 % burner efficiency

def night_mask(wx):
    h = wx.hour[wx.n_spin:]
    return (h >= 18) | (h < 6)

def cost(x, mat, sh=Shelter()):
    """Build cost in INR lakh."""
    _, _, Aw, Ae, _ = geom(x, sh)
    return (Ae * (x[0] * MATERIALS[mat].price + x[1] * PRICE_INS + x[2] * PRICE_PCM) + Aw * PRICE_WIN) / 1e5

def hard_ncr(Ti, mask):
    Ti = np.asarray(Ti)[mask]
    return float(np.mean((Ti >= BAND[0]) & (Ti <= BAND[1])))

def soft_ncr(Ti, mask, s=0.7):
    """Smooth NCR for gradient search."""
    m = xp.asarray(mask.astype(float))
    sig = lambda z: 1 / (1 + xp.exp(-z))
    return xp.sum(sig((Ti - BAND[0]) / s) * sig((BAND[1] - Ti) / s) * m) / mask.sum()

def deficit(Ti, mask):
    """Mean shortfall below the comfort band at night (C) - keeps a gradient when far from comfort."""
    return xp.sum(xp.maximum(0.0, BAND[0] - Ti) * xp.asarray(mask.astype(float))) / mask.sum()

def summarize(x, wx, mat, sh=Shelter()):
    """All headline numbers for one design at one site. Energies are per day."""
    x = np.asarray(x, float)
    free = make_sim(wx, mat, False, sh)(x)
    hot = make_sim(wx, mat, True, sh)(x)
    base = make_sim(wx, mat, True, sh)(BASELINE)
    mask, days = night_mask(wx), len(wx.Ta[wx.n_spin:]) * DT / 86400
    kwh = lambda r: float(np.sum(r["Qh"]) * DT / 3.6e6 / days)
    Ti = np.asarray(free["Ti"])
    out = dict(ncr=hard_ncr(Ti, mask), tmin_night=float(Ti[mask].min()),
               heat_kwh=kwh(hot), fuel_kg=kwh(hot) / KWH_PER_KG,
               base_fuel_kg=kwh(base) / KWH_PER_KG,
               solar_kwh=float(np.sum(free["Qs"]) * DT / 3.6e6 / days), cost_lakh=float(cost(x, mat, sh)))
    out["hot_frac"] = float(np.mean(Ti > BAND[1]))          # share of the day above the comfort band
    out["fuel_saved_kg"] = out["base_fuel_kg"] - out["fuel_kg"]
    out["series"] = {k: np.asarray(v) for k, v in free.items()}
    out["hour"] = wx.hour[wx.n_spin:]
    out["passive_ok"] = out["ncr"] >= 0.9
    return out
