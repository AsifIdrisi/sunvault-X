"""Two-node RC thermal model of a passive shelter (F4, F5, F11).
Uses JAX (autodiff + jit) when installed, otherwise NumPy with finite-difference gradients.

  Ca dTi/dt = Qsolar/2 + Qpeople - (Ti-Tw)/Rin - (Ti-Ta)*Gd + Qheater
  Cw(Tw) dTw/dt = (Ti-Tw)/Rin - (Tw-Ta)/Rout + Qsolar/2
Cw(Tw) includes a smoothed-enthalpy (Gaussian) PCM latent peak so gradients stay stable.
"""
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from .materials import MATERIALS, K_INS, C_PCM, L_PCM, T_MELT, W_MELT, U_WIN

try:
    import jax, jax.numpy as xp
    from jax import lax
    jax.config.update("jax_enable_x64", True)
    HAS_JAX = True
except ImportError:
    xp, HAS_JAX = np, False

DT = 300.0                    # model step, s
H_IN, H_OUT = 8.0, 20.0       # surface film coefficients W/m2K
C_FURN = 4e5                  # furniture / contents heat capacity J/K
G_WIN = 0.6                   # window solar transmittance
NAMES = ["thickness", "insulation", "pcm", "window", "orientation", "aspect"]
LABELS = ["Wall thickness (m)", "Insulation (m)", "PCM layer (m)", "Window ratio (south wall)",
          "Orientation off south (deg)", "Plan aspect (long/short)"]
LO = np.array([0.10, 0.00, 0.00, 0.05, 0.0, 1.0])
HI = np.array([0.60, 0.15, 0.05, 0.50, 90.0, 3.0])
DEFAULT = np.array([0.30, 0.05, 0.00, 0.25, 20.0, 1.5])
BASELINE = np.array([0.15, 0.00, 0.00, 0.20, 45.0, 1.0])   # "conventional" shelter for fuel-saved

@dataclass(frozen=True)
class Shelter:
    floor: float = 16.0       # m2
    height: float = 2.6       # m
    occupants: int = 5
    q_person: float = 75.0    # W
    setpoint: float = 18.0    # C, heating hold temperature

def geom(x, sh):
    """Plan is L x W with the long side facing (near) south. Returns L, W, window area, opaque envelope area, volume."""
    L = xp.sqrt(sh.floor * x[5]); W = sh.floor / L
    Aw = x[3] * L * sh.height
    return L, W, Aw, 2 * (L + W) * sh.height + sh.floor - Aw, sh.floor * sh.height

def _scan(f, init, xs):
    if HAS_JAX:
        return lax.scan(f, init, xs)
    carry, ys = init, []
    for i in range(len(xs[0])):
        carry, y = f(carry, tuple(a[i] for a in xs))
        ys.append(y)
    return carry, tuple(np.array([y[j] for y in ys]) for j in range(len(ys[0])))

@lru_cache(maxsize=64)
def make_sim(wx, mat, heat=False, shelter=Shelter()):
    """Return sim(x, kR=1, ach=0.2) -> dict of arrays (spin-up removed). kR scales wall resistance, ach = air changes/h."""
    m, Ta, G, n0 = MATERIALS[mat], xp.asarray(wx.Ta), xp.asarray(wx.G), wx.n_spin
    def sim(x, kR=1.0, ach=0.2):
        t, ins, pcm, ori = x[0], x[1], x[2], x[4]
        _, _, Aw, Ae, V = geom(x, shelter)
        Rw = kR * (t / m.k + ins / K_INS) / Ae
        Rin, Rout = 1 / (H_IN * Ae) + 0.5 * Rw, 1 / (H_OUT * Ae) + 0.5 * Rw
        Gd = U_WIN * Aw + ach * V * 0.34
        Ca, Cw0 = V * 1200 + C_FURN, Ae * t * m.c
        ofac = xp.maximum(0.15, xp.cos(ori * np.pi / 180))
        Qp = shelter.occupants * shelter.q_person
        def step(c, inp):
            Ti, Tw = c; ta, g = inp
            Qs = g * Aw * G_WIN * ofac
            z = (Tw - T_MELT) / W_MELT
            Cw = Cw0 + Ae * pcm * (C_PCM + L_PCM * xp.exp(-z * z) / (W_MELT * 1.7724539))
            Qiw, Qwo, Qd = (Ti - Tw) / Rin, (Tw - ta) / Rout, (Ti - ta) * Gd
            Tf = Ti + DT / Ca * (Qs / 2 + Qp - Qiw - Qd)
            Qh = xp.maximum(0.0, shelter.setpoint - Tf) * Ca / DT if heat else 0.0 * Tf
            return (Tf + Qh * DT / Ca, Tw + DT / Cw * (Qiw - Qwo + Qs / 2)), (Tf + Qh * DT / Ca, Tw, Qs, Qd + Qwo, Qh)
        T0 = xp.mean(Ta)
        _, ys = _scan(step, (T0, T0), (Ta, G))
        Ti, Tw, Qs, Ql, Qh = (y[n0:] for y in ys)
        return dict(Ti=Ti, Tw=Tw, Qs=Qs, Ql=Ql, Qh=Qh)
    return jax.jit(sim) if HAS_JAX else sim
