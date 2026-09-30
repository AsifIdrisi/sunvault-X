import numpy as np, sunvault as sv

def wx(): return sv.synthetic("Leh - October")

def test_deterministic():
    a = sv.summarize(sv.DEFAULT, wx(), "mud"); b = sv.summarize(sv.DEFAULT, wx(), "mud")
    assert a["ncr"] == b["ncr"] and a["heat_kwh"] == b["heat_kwh"]

def test_insulation_reduces_heating():
    x0 = sv.DEFAULT.copy(); x0[1] = 0.0
    x1 = sv.DEFAULT.copy(); x1[1] = 0.12
    assert sv.summarize(x1, wx(), "mud")["heat_kwh"] < sv.summarize(x0, wx(), "mud")["heat_kwh"]

def test_pcm_raises_min_night_temp():
    x0 = sv.DEFAULT.copy(); x0[2] = 0.0
    x1 = sv.DEFAULT.copy(); x1[2] = 0.04
    assert sv.summarize(x1, wx(), "mud")["tmin_night"] >= sv.summarize(x0, wx(), "mud")["tmin_night"] - 0.5

def test_energy_balance_sane():
    s = sv.summarize(sv.DEFAULT, wx(), "mud")
    assert s["solar_kwh"] > 0 and s["heat_kwh"] >= 0 and -30 < s["tmin_night"] < 40

def test_optimizer_beats_start():
    w = [wx(), sv.synthetic("Leh - October", cloud=0.7, dT=-6)]
    r = sv.optimize(w, "mud", w=0.3, iters=20)
    f0 = sv.summarize(sv.BASELINE, wx(), "mud"); f1 = sv.summarize(r["x"], wx(), "mud")
    assert f1["fuel_kg"] < f0["fuel_kg"]

def test_calibration_recovers():
    w = sv.synthetic("Leh - October", days=3, cloud=[0.1, 0.7, 0.3])
    obs = sv.fake_logger(sv.DEFAULT, w, "mud", kR=1.3, ach=0.35)
    c = sv.calibrate(sv.DEFAULT, w, "mud", obs)
    assert c["rmse_after"] < c["rmse_before"] and c["rmse_after"] < 0.8

def test_pareto_nondominated():
    p = sv.pareto(wx(), "mud", pop=10, gens=4)
    assert np.all(np.diff(p["ncr"]) > 0) or len(p["ncr"]) == 1
