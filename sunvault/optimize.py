"""Inverse design (F7), Pareto set (F8) and worst-case design (F9)."""
import numpy as np
from .model import HAS_JAX, LO, HI, DEFAULT, Shelter, make_sim, xp
from .metrics import soft_ncr, deficit, hard_ncr, night_mask, cost
if HAS_JAX:
    import jax

def _objective(wxs, mat, w, sh):
    sims = [make_sim(wx, mat, False, sh) for wx in wxs]
    mask = night_mask(wxs[0])
    def f(z):
        x = LO + z * (HI - LO)
        js = xp.stack([soft_ncr(s(x)["Ti"], mask) - 0.02 * deficit(s(x)["Ti"], mask) for s in sims])
        J = -0.05 * xp.log(xp.sum(xp.exp(-js / 0.05)))           # soft-min: worst scenario dominates
        return J - w * 0.15 * cost(x, mat, sh)
    return f

def optimize(wxs, mat, w=0.5, sh=Shelter(), iters=50, x0=None):
    """Adam ascent on (smooth NCR - cost penalty). wxs = list of weather scenarios (normal, stress...).
    w in [0,1]: 0 = comfort first, 1 = cost first."""
    f = _objective(wxs if isinstance(wxs, (list, tuple)) else [wxs], mat, w, sh)
    if HAS_JAX:
        vg = jax.jit(jax.value_and_grad(f))
        vg_ = lambda z: (float(vg(z)[0]), np.asarray(vg(z)[1]))
    else:
        def vg_(z, e=0.02):
            j0 = float(f(z)); g = np.zeros_like(z)
            for i in range(len(z)):
                zz = z.copy(); zz[i] += e; g[i] = (float(f(zz)) - j0) / e
            return j0, g
    z_start = (np.asarray(x0 if x0 is not None else DEFAULT) - LO) / (HI - LO)
    best = (-1e9, z_start, [])
    for z in (z_start, np.array([.5, .5, .5, .5, .2, .3]), np.array([.8, .9, .9, .6, .1, .6])):
        z, m, v, hist = z.copy(), 0, 0, []
        for i in range(1, iters + 1):
            j, g = vg_(z); hist.append(j)
            m, v = 0.9 * m + 0.1 * g, 0.999 * v + 0.001 * g * g
            z = np.clip(z + 0.06 * (m / (1 - 0.9 ** i)) / (np.sqrt(v / (1 - 0.999 ** i)) + 1e-8), 0, 1)
        j = vg_(z)[0]
        if j > best[0]: best = (j, z, hist)
    return dict(x=LO + best[1] * (HI - LO), J=best[0], history=best[2])

def _nondominated(ncr, c):
    idx = np.argsort(c); keep, top = [], -1
    for i in idx:
        if ncr[i] > top: keep.append(i); top = ncr[i]
    return np.array(keep)

def pareto(wx, mat, sh=Shelter(), pop=40, gens=25, seed=1):
    """Cost vs NCR trade-off set. Uses pymoo NSGA-II if installed, else random search."""
    sim, mask = make_sim(wx, mat, False, sh), night_mask(wx)
    ev = lambda x: (hard_ncr(sim(x)["Ti"], mask), float(cost(x, mat, sh)))
    try:
        from pymoo.core.problem import ElementwiseProblem
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.optimize import minimize
        class P(ElementwiseProblem):
            def __init__(s): super().__init__(n_var=6, n_obj=2, xl=LO, xu=HI)
            def _evaluate(s, x, out, *a, **k):
                n, c = ev(x); out["F"] = [-n, c]
        r = minimize(P(), NSGA2(pop_size=pop), ("n_gen", gens), seed=seed, verbose=False)
        X = np.atleast_2d(r.X)
    except ImportError:
        X = LO + np.random.default_rng(seed).random((pop * gens // 4, 6)) * (HI - LO)
    F = np.array([ev(x) for x in X])
    k = _nondominated(F[:, 0], F[:, 1])
    return dict(X=X[k], ncr=F[k, 0], cost=F[k, 1])
