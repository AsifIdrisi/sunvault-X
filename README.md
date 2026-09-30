# SunVault — Flask passive shelter designer

SunVault is a Python web dashboard for simulating shelter temperature, estimating night comfort and heating demand, testing cold/cloudy scenarios, finding design parameters, calibrating the model against logger data, and downloading a design report.

## Run locally

```bash
# From the sunvault_project folder
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:5000** in Chrome. Keep the terminal running while using the app. Stop it with `Ctrl+C` when finished.

## Features
- Flask-based responsive web dashboard (no Streamlit required)
- Weather presets and optional hourly weather CSV upload (`temp_c`/`T2M`, `ghi_wm2`/`ALLSKY_SFC_SW_DWN`)
- Wall-material selection and custom material addition
- Shelter design sliders, comfort/temperature/heating/cost metrics, and plots
- Cold and cloudy stress testing
- Auto-design optimisation
- Model calibration from a logger CSV (`Ti`) or demo data

## Climate presets
Cold: Leh (Oct/Jan), Tawang, Srinagar, Shimla. Hot: Jaisalmer (May), Gorakhpur (June). Presets are synthetic; upload real hourly NASA POWER/IMD data for a site.

## Tests

```bash
python -m pytest tests
```

## Important limitations
Default weather is synthetic. Model constants and material properties are assumptions and have not necessarily been validated for a real shelter. Treat the output as design-support estimates, not a safety certification.
# sunvault-X
