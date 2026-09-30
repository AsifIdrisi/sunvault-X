"""Report and ANSYS-ready case export (F12)."""
import json, os
import numpy as np
from .model import NAMES, Shelter, geom
from .materials import MATERIALS, K_INS, U_WIN

def markdown_report(site, mat, x, s, stress=None):
    rows = "\n".join(f"| {n} | {v:.3f} |" for n, v in zip(NAMES, x))
    verdict = ("Comfortable overnight without fuel." if s["passive_ok"] else
               f"Passive comfort NOT reached. Minimum heating to hold 18 C: {s['heat_kwh']:.1f} kWh/day ({s['fuel_kg']:.1f} kg fuel).")
    st = f"\n- Night Comfort Ratio in stress test: {stress['ncr']*100:.0f} %" if stress else ""
    return f"""# SunVault design report

**Site:** {site}  |  **Wall material:** {MATERIALS[mat].name}

| Design variable | Value |
|---|---|
{rows}

## Results
- Night Comfort Ratio (18-26 C, 18:00-06:00): **{s['ncr']*100:.0f} %**
- Coldest night temperature: {s['tmin_night']:.1f} C
- Heating to hold 18 C: {s['heat_kwh']:.1f} kWh/day = {s['fuel_kg']:.1f} kg fuel/day
- Fuel saved vs conventional shelter: {s['fuel_saved_kg']:.1f} kg/day
- Solar energy gained: {s['solar_kwh']:.1f} kWh/day
- Build cost: INR {s['cost_lakh']:.2f} lakh{st}

**{verdict}**

## Assumptions and limits
Two-node RC model; one lumped wall node; synthetic or supplied hourly weather; 0.2 air changes/h unless calibrated;
no CFD, structural or fire-safety checks. Validate finalists in ANSYS/EnergyPlus before use.
"""

def write_ansys_case(folder, site, mat, x, wx, sh=Shelter()):
    """Write case.json (geometry, layers, material properties) and boundary.csv (hourly ambient + solar)."""
    os.makedirs(folder, exist_ok=True)
    L, W, Aw, _, _ = (float(v) for v in geom(np.asarray(x, float), sh))
    m = MATERIALS[mat]
    case = dict(site=site, geometry=dict(length_m=L, width_m=W, height_m=sh.height, window_area_m2=Aw,
                orientation_deg_off_south=float(x[4])),
                layers=[dict(name=m.name, thickness_m=float(x[0]), k=m.k, volumetric_heat_capacity=m.c),
                        dict(name="insulation", thickness_m=float(x[1]), k=K_INS),
                        dict(name="PCM", thickness_m=float(x[2]), melt_c=20.0)],
                window_u=U_WIN, internal_gain_w=sh.occupants * sh.q_person, air_changes_per_h=0.2)
    json.dump(case, open(os.path.join(folder, "case.json"), "w"), indent=2)
    with open(os.path.join(folder, "boundary.csv"), "w") as f:
        f.write("hour,ambient_c,solar_wm2\n")
        for i, (a, g) in enumerate(zip(wx.Ta_h, wx.G_h)): f.write(f"{i},{a:.2f},{g:.1f}\n")
    return folder
