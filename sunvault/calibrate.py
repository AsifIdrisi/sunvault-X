"""Self-calibration (F10): fit wall-resistance multiplier and air changes/h from a few days of logger data.
MAP estimate with Gaussian priors + Laplace approximation -> posterior standard deviations."""
import numpy as np
from scipy.optimize import least_squares
from .model import make_sim, Shelter

PRIOR_MU = np.log([1.0, 0.2]); PRIOR_SD = np.array([0.5, 0.7])   # log-scale priors

def fake_logger(x, wx, mat, kR=1.3, ach=0.35, noise=0.3, seed=0, sh=Shelter()):
    """Demo logger: 'true' shelter that differs from the assumed one, plus sensor noise (hourly Ti)."""
    Ti = np.asarray(make_sim(wx, mat, False, sh)(np.asarray(x, float), kR, ach)["Ti"])[::12]
    return Ti + np.random.default_rng(seed).normal(0, noise, Ti.shape)

def calibrate(x, wx, mat, Ti_obs, sigma=0.5, sh=Shelter()):
    sim, x = make_sim(wx, mat, False, sh), np.asarray(x, float)
    hourly = lambda th: np.asarray(sim(x, np.exp(th[0]), np.exp(th[1]))["Ti"])[::12]
    Ti_obs = np.asarray(Ti_obs)[:len(hourly(PRIOR_MU))]
    res = lambda th: np.concatenate([(hourly(th)[:len(Ti_obs)] - Ti_obs) / sigma, (th - PRIOR_MU) / PRIOR_SD])
    r = least_squares(res, PRIOR_MU)
    cov = np.linalg.inv(r.jac.T @ r.jac)
    sd = np.sqrt(np.diag(cov))
    rmse = lambda th: float(np.sqrt(np.mean((hourly(th)[:len(Ti_obs)] - Ti_obs) ** 2)))
    th = r.x
    return dict(kR=float(np.exp(th[0])), ach=float(np.exp(th[1])),
                kR_range=(float(np.exp(th[0] - 2 * sd[0])), float(np.exp(th[0] + 2 * sd[0]))),
                ach_range=(float(np.exp(th[1] - 2 * sd[1])), float(np.exp(th[1] + 2 * sd[1]))),
                rmse_before=rmse(PRIOR_MU), rmse_after=rmse(th))
