"""Material library (F2). k in W/mK, c = volumetric heat capacity J/m3K, price in INR per m3."""
from dataclasses import dataclass

@dataclass(frozen=True)
class Material:
    name: str
    k: float
    c: float
    price: float

MATERIALS = {
    "mud":      Material("Mud / adobe", 0.6, 1.5e6, 2000),
    "stone":    Material("Stone", 2.0, 2.0e6, 5000),
    "concrete": Material("Concrete", 1.4, 2.0e6, 7000),
    "wood":     Material("Wood", 0.15, 0.8e6, 12000),
}

# Fixed layers (used for every design)
K_INS, PRICE_INS = 0.035, 6000      # mineral wool / EPS-like insulation
C_PCM, L_PCM, PRICE_PCM = 1.8e6, 1.2e8, 90000   # PCM sensible + latent (150 kJ/kg * 800 kg/m3)
T_MELT, W_MELT = 20.0, 2.0          # melt centre and smoothing width (C)
U_WIN, PRICE_WIN = 2.8, 8000        # double glazing, W/m2K and INR/m2

def add_material(key, name, k, c, price):
    """User-defined material."""
    MATERIALS[key] = Material(name, k, c, price)
