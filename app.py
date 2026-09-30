"""SunVault Flask dashboard. Run with: python app.py"""
import io, os, base64
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, request, session, render_template_string, flash, redirect, url_for
import sunvault as sv
from sunvault.model import LABELS
from flask import Response

app = Flask(__name__)
app.secret_key = os.environ.get('SUNVAULT_SECRET_KEY', 'change-this-dev-secret')

DEFAULT_DESIGN = [float(v) for v in sv.DEFAULT]
HTML = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="theme-color" content="#17352b"><title>SunVault | Passive Shelter Design</title><style>
:root{--bg:#f4f1ea;--card:#fffdf8;--ink:#1d2a24;--muted:#6b766f;--brand:#c2571a;--brand2:#e8a33d;--deep:#17352b;--line:#e6dfd0;--ok:#1f7a55;--okbg:#e3f3ea;--warnbg:#fdeccd;--warn:#7a4b00;--soft:#f6f1e5}
@media(prefers-color-scheme:dark){:root{--bg:#121a17;--card:#1a2420;--ink:#ece8dd;--muted:#98a59d;--line:#2c3a34;--soft:#212e28;--okbg:#173a2b;--ok:#7fd8ac;--warnbg:#3d2d0c;--warn:#f1c675}}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--bg);font:15px/1.55 "Segoe UI",system-ui,-apple-system,sans-serif;color:var(--ink)}
header{background:linear-gradient(115deg,var(--deep),#2b5a47 58%,#8a4a1c);color:#fff;padding:17px max(16px,calc((100% - 1220px)/2));box-shadow:0 5px 22px #00000018}.brand-row{display:flex;align-items:center;justify-content:space-between;gap:16px}.brand{display:flex;align-items:center;gap:12px}header h1{margin:0;font-size:30px;letter-spacing:-.7px}header h1 span{color:var(--brand2)}.header-sub{margin:2px 0 0;color:#d5e4dc;font-size:13px}
.menu-btn{width:44px;height:44px;border:1px solid rgba(255,255,255,.3);background:rgba(255,255,255,.1);color:#fff;border-radius:12px;font-size:26px;line-height:1;cursor:pointer}.menu-btn:hover{background:rgba(255,255,255,.18)}
.menu-overlay{position:fixed;inset:0;background:#0007;z-index:1000;opacity:0;visibility:hidden;transition:opacity .2s,visibility .2s}.menu-overlay.open{opacity:1;visibility:visible}.menu-panel{position:absolute;right:0;top:0;height:100%;width:min(410px,94vw);background:var(--card);color:var(--ink);box-shadow:-12px 0 35px #0004;overflow:auto;padding:20px 17px}.menu-head{display:flex;align-items:center;justify-content:space-between;margin-bottom:14px;position:sticky;top:-20px;background:var(--card);padding:4px 0 12px;z-index:2}.menu-head h2{margin:0;font-size:20px}.menu-head p{margin:2px 0 0;color:var(--muted);font-size:12px}.close-btn{border:0;background:var(--soft);color:var(--ink);width:38px;height:38px;border-radius:10px;font-size:22px;cursor:pointer}
main{max-width:1220px;margin:22px auto 28px;padding:0 16px}.layout{display:block}.card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:20px;margin-bottom:18px;box-shadow:0 6px 22px #0000000d}.card h2{margin:0 0 4px;font-size:20px}.card h3{margin:18px 0 8px;font-size:15px}.muted{color:var(--muted);font-size:12.5px}.field{margin:12px 0}.field label{display:flex;justify-content:space-between;font-weight:600;font-size:13px;margin-bottom:5px}.field input,.field select{width:100%;padding:9px 10px;border:1px solid var(--line);border-radius:9px;background:var(--soft);color:var(--ink)}input[type=range]{padding:0;accent-color:var(--brand);height:6px}.btn{display:inline-block;border:0;border-radius:10px;padding:10px 15px;background:var(--brand);color:#fff;font-weight:650;cursor:pointer;text-decoration:none;font-size:14px}.btn:hover{filter:brightness(1.08)}.btn.secondary{background:var(--soft);color:var(--ink);border:1px solid var(--line)}.btn.green{background:var(--ok);color:#fff}.actions{display:flex;gap:9px;flex-wrap:wrap;margin-top:14px}.zone{background:var(--soft);border:1px dashed var(--line);border-radius:10px;padding:8px 11px;font-size:12.5px;margin-top:6px}
.metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-top:14px}.metric{background:var(--soft);border:1px solid var(--line);border-radius:12px;padding:13px;border-top:3px solid var(--brand2)}.metric .label{font-size:11.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.4px}.metric .value{font-size:22px;font-weight:750;margin-top:3px}.metric small{color:var(--muted)}.status{padding:12px 14px;border-radius:10px;margin:14px 0;font-weight:550}.good{background:var(--okbg);color:var(--ok)}.warn{background:var(--warnbg);color:var(--warn)}.tips{background:var(--soft);border:1px solid var(--line);border-radius:12px;padding:6px 18px 10px}.tips li{margin:7px 0}
.tabs{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:16px}.tabs a{padding:9px 14px;border-radius:999px;background:var(--card);border:1px solid var(--line);color:var(--muted);text-decoration:none;font-weight:650}.tabs a.active{background:var(--deep);color:#fff;border-color:var(--deep)}.plot{width:100%;height:auto;border:1px solid var(--line);border-radius:12px;margin:6px 0 10px;background:#fff}.flash{background:var(--warnbg);color:var(--warn);padding:11px 14px;border-radius:10px;margin-bottom:12px}.table{width:100%;border-collapse:collapse}.table td,.table th{text-align:left;padding:8px;border-bottom:1px solid var(--line)}footer{text-align:center;color:var(--muted);padding:8px 16px 30px;font-size:13px}
@media(max-width:900px){.metrics{grid-template-columns:repeat(2,minmax(0,1fr))}.card{padding:16px}}
@media(max-width:600px){header{padding:14px}.brand-row{gap:10px}header h1{font-size:26px}.header-sub{font-size:12px}.menu-btn{width:42px;height:42px}main{padding:0 10px;margin:14px auto 22px}.metrics{grid-template-columns:1fr}.tabs{overflow-x:auto;flex-wrap:nowrap;padding-bottom:3px}.tabs a{white-space:nowrap}.card{border-radius:13px}.actions .btn{width:100%;text-align:center}.plot{border-radius:9px}.table{display:block;overflow-x:auto;white-space:nowrap}}
</style></head><body>
<header><div class="brand-row"><div class="brand"><div><h1>Sun<span>Vault</span></h1><p class="header-sub">Passive shelter design & thermal comfort analysis</p></div></div><button class="menu-btn" type="button" aria-label="Open design menu" aria-controls="appMenu" aria-expanded="false" onclick="openMenu()">⋮</button></div></header>
<div class="menu-overlay" id="appMenu" onclick="if(event.target===this) closeMenu()"><aside class="menu-panel" aria-label="Design controls"><div class="menu-head"><div><h2>Design workspace</h2><p>Configure the shelter model</p></div><button class="close-btn" type="button" onclick="closeMenu()" aria-label="Close menu">×</button></div><form method="post" enctype="multipart/form-data" class="card"><h2>Design controls</h2><input type="hidden" name="action" value="simulate">
<div class="field"><label for="site">Weather preset</label><select name="site" id="site">{% for key in sites %}<option value="{{key}}" {{'selected' if key==site else ''}}>{{key}}</option>{% endfor %}</select></div>
<div class="zone" style="margin-top:-6px">Climate zone: <strong>{{zone}}</strong></div>
<div class="field"><label for="mat">Wall material</label><select name="mat" id="mat">{% for key,m in materials.items() %}<option value="{{key}}" {{'selected' if key==mat else ''}}>{{m.name}}</option>{% endfor %}</select></div>
<div class="field"><label>Optional hourly weather CSV</label><input type="file" name="weather_csv" accept=".csv"><div class="muted">Columns: temp_c or T2M, and ghi_wm2 or ALLSKY_SFC_SW_DWN. CSV is used for this request.</div></div>
{% for i in range(names|length) %}<div class="field"><label for="v{{i}}">{{labels[i]}} <span class="muted" id="val{{i}}">{{'%.3f'|format(design[i])}}</span></label><input type="range" id="v{{i}}" name="v{{i}}" min="{{lo[i]}}" max="{{hi[i]}}" step="{{(hi[i]-lo[i])/200}}" value="{{design[i]}}" oninput="document.getElementById('val{{i}}').textContent=(+this.value).toFixed(3)"></div>{% endfor %}
<div class="actions"><button class="btn" type="submit">Analyze design</button><button class="btn secondary" type="submit" formaction="{{url_for('reset')}}" formmethod="post">Reset defaults</button></div></form>
<div class="card"><h3>Material library</h3><form method="post"><input type="hidden" name="action" value="add_material"><div class="field"><label>Name</label><input name="material_name" required placeholder="e.g. Local brick"></div><div class="field"><label>Conductivity (W/mK)</label><input name="conductivity" type="number" min="0.02" max="5" step="0.01" value="0.5" required></div><div class="field"><label>Heat capacity (MJ/m³K)</label><input name="capacity" type="number" min="0.1" max="4" step="0.1" value="1.5" required></div><div class="field"><label>Price (INR/m³)</label><input name="price" type="number" min="100" max="100000" value="3000" required></div><button class="btn secondary" type="submit">Add material</button></form></div></aside></div>
<section>
</aside></div>
<main><div class="layout">
{% with messages = get_flashed_messages() %}{% for message in messages %}<div class="flash">{{message}}</div>{% endfor %}{% endwith %}
<nav class="tabs">{% for key,label in [('simulate','Analysis'),('stress','Stress test'),('optimise','Optimize design'),('calibrate','Calibration')] %}<a class="{{'active' if page==key else ''}}" href="{{url_for('index',page=key)}}">{{label}}</a>{% endfor %}</nav>
{% if page=='simulate' %}<div class="card"><h2>Simulation results · {{site}}</h2><div class="metrics">
<div class="metric"><div class="label">Night Comfort Ratio</div><div class="value">{{'%.0f'|format(S.ncr*100)}}%</div></div><div class="metric"><div class="label">Coldest night</div><div class="value">{{'%.1f'|format(S.tmin_night)}} °C</div></div><div class="metric"><div class="label">Heating to hold 18 °C</div><div class="value">{{'%.1f'|format(S.heat_kwh)}}</div><small>kWh/day · {{'%.1f'|format(S.fuel_kg)}} kg fuel</small></div>
<div class="metric"><div class="label">Estimated build cost</div><div class="value">₹{{'%.2f'|format(S.cost_lakh)}} L</div></div><div class="metric"><div class="label">Fuel saved vs baseline</div><div class="value">{{'%.1f'|format(S.fuel_saved_kg)}} kg/day</div></div><div class="metric"><div class="label">Solar energy gained</div><div class="value">{{'%.1f'|format(S.solar_kwh)}} kWh/day</div></div>
<div class="metric"><div class="label">Time above 26 °C</div><div class="value">{{'%.0f'|format(S.hot_frac*100)}}%</div><small>overheating risk</small></div><div class="metric"><div class="label">Stress-test NCR</div><div class="value">{{'%.0f'|format(SS.ncr*100)}}%</div><small>cold + cloudy</small></div></div>
<div class="status {{'good' if S.passive_ok else 'warn'}}">{{'Comfortable overnight without fuel under this model.' if S.passive_ok else 'Passive comfort target not reached. Estimated minimum heating to hold 18 °C: %.1f kWh/day.'|format(S.heat_kwh)}}</div>
<h3>Design guidance for this site</h3><ul class="tips">{% for t in tips %}<li>{{t}}</li>{% endfor %}</ul><div class="actions"><a class="btn secondary" href="{{url_for('report')}}">⬇ Download design report</a></div>
<h3>Inside vs outside temperature</h3><img class="plot" src="data:image/png;base64,{{plots.temp}}" alt="Temperature chart"><h3>Solar gain vs heat loss</h3><img class="plot" src="data:image/png;base64,{{plots.heat}}" alt="Heat flow chart"><h3>Temperature difference vs heat loss</h3><img class="plot" src="data:image/png;base64,{{plots.scatter}}" alt="Heat loss scatter chart"></div>
{% elif page=='stress' %}<div class="card"><h2>Cold and cloudy stress test</h2><div class="metrics"><div class="metric"><div class="label">Normal NCR</div><div class="value">{{'%.0f'|format(S.ncr*100)}}%</div></div><div class="metric"><div class="label">Stress NCR</div><div class="value">{{'%.0f'|format(SS.ncr*100)}}%</div></div><div class="metric"><div class="label">Extra heating in stress</div><div class="value">{{'%.1f'|format(SS.heat_kwh-S.heat_kwh)}} kWh/day</div></div></div><img class="plot" src="data:image/png;base64,{{plots.stress}}" alt="Stress test chart"><p class="muted">Stress scenario: cloud cover 70% and outside temperature shifted down by 6 °C.</p></div>
{% elif page=='optimise' %}<div class="card"><h2>Auto-design</h2><p>Search for a shelter design that balances night comfort and build cost.</p><form method="post"><input type="hidden" name="action" value="optimise"><div class="field"><label>Comfort first ↔ cost first (0–1)</label><input type="range" name="weight" min="0" max="1" step="0.05" value="0.3"></div><label><input type="checkbox" name="worst" checked> Include cold/cloudy stress scenario</label><div class="actions"><button class="btn green">Find a design</button></div></form>{% if opt %}<div class="status good">Optimisation complete. The suggested design has been loaded into the sliders; return to Simulate to review results.</div><table class="table"><tr><th>Parameter</th><th>Suggested value</th></tr>{% for i in range(names|length) %}<tr><td>{{labels[i]}}</td><td>{{'%.4f'|format(opt[i])}}</td></tr>{% endfor %}</table>{% endif %}</div>
{% elif page=='calibrate' %}<div class="card"><h2>Model calibration</h2><p>Upload a logger CSV containing an hourly <code>Ti</code> column, or run a demonstration using simulated sensor data.</p><form method="post" enctype="multipart/form-data"><input type="hidden" name="action" value="calibrate"><div class="field"><label>Logger CSV (optional)</label><input type="file" name="logger_csv" accept=".csv"></div><button class="btn green">Run calibration</button></form>{% if calibration %}<div class="metrics" style="margin-top:16px"><div class="metric"><div class="label">RMSE before → after</div><div class="value">{{'%.2f'|format(calibration.rmse_before)}} → {{'%.2f'|format(calibration.rmse_after)}} °C</div></div><div class="metric"><div class="label">Wall resistance multiplier</div><div class="value">{{'%.2f'|format(calibration.kR)}}</div><small>95% range {{'%.2f'|format(calibration.kR_range[0])}}–{{'%.2f'|format(calibration.kR_range[1])}}</small></div><div class="metric"><div class="label">Air changes / hour</div><div class="value">{{'%.2f'|format(calibration.ach)}}</div><small>95% range {{'%.2f'|format(calibration.ach_range[0])}}–{{'%.2f'|format(calibration.ach_range[1])}}</small></div></div>{% endif %}</div>
{% endif %}
</section></div></main>
<footer><strong>SunVault</strong> · Passive shelter design workspace</footer>
<script>function openMenu(){const m=document.getElementById('appMenu');m.classList.add('open');document.querySelector('.menu-btn').setAttribute('aria-expanded','true');document.body.style.overflow='hidden'}function closeMenu(){const m=document.getElementById('appMenu');m.classList.remove('open');document.querySelector('.menu-btn').setAttribute('aria-expanded','false');document.body.style.overflow=''}document.addEventListener('keydown',e=>{if(e.key==='Escape')closeMenu()});</script></body></html>'''


def current_design():
    vals = session.get('design', DEFAULT_DESIGN)
    return np.clip(np.asarray(vals, dtype=float), sv.LO, sv.HI)

def get_weather(site, uploaded=None):
    if uploaded and uploaded.filename:
        return sv.from_csv(uploaded)
    return sv.synthetic(site)

def fig_b64(fig):
    buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format='png', dpi=120, bbox_inches='tight'); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('ascii')

def make_plots(S, wx, SS=None, wx_stress=None):
    plots = {}
    h = np.arange(len(S['series']['Ti'])) * sv.model.DT / 3600
    fig, ax = plt.subplots(figsize=(9, 3.4)); ax.axhspan(18, 26, color='green', alpha=.12, label='comfort band')
    ax.plot(h, S['series']['Ti'], lw=2, label='inside', color='#e58a00'); ax.plot(h, wx.Ta[wx.n_spin:], lw=1.5, label='outside', color='#2f6fd6')
    ax.set(xlabel='Hours', ylabel='Temperature (°C)', title='Inside temperature'); ax.legend(); plots['temp'] = fig_b64(fig)
    ser = S['series']; hh = np.arange(len(ser['Qs'])) * sv.model.DT / 3600
    fig, ax = plt.subplots(figsize=(9, 3.2)); ax.plot(hh, ser['Qs'], label='solar gain (W)', color='#e58a00'); ax.plot(hh, ser['Ql'], label='heat loss (W)', color='#2f6fd6'); ax.set(xlabel='Hours', ylabel='Power (W)'); ax.legend(); plots['heat'] = fig_b64(fig)
    fig, ax = plt.subplots(figsize=(9, 3.2)); ax.scatter(ser['Ti'] - wx.Ta[wx.n_spin:], ser['Ql'], s=7); ax.set(xlabel='Inside − outside (°C)', ylabel='Heat loss (W)', title='Heat loss relationship'); plots['scatter'] = fig_b64(fig)
    if SS is not None:
        fig, ax = plt.subplots(figsize=(9, 3.2)); ax.axhspan(18, 26, color='green', alpha=.12); ax.plot(SS['series']['Ti'], label='inside (stress)', color='#e58a00'); ax.plot(wx_stress.Ta[wx_stress.n_spin:], label='outside (stress)', color='#2f6fd6'); ax.set(xlabel='Simulation step', ylabel='Temperature (°C)'); ax.legend(); plots['stress'] = fig_b64(fig)
    return plots

def advice(S, SS, x, site):
    """Rule-based, plain-language guidance from the simulated results."""
    t, ins, pcm, win, ori, _ = x
    hot_site = sv.SITES[site]['m'] > 25
    tips = []
    if hot_site:
        if S['hot_frac'] > 0.3: tips.append('High overheating: reduce the window ratio, thicken the walls (thermal mass) and add insulation to slow daytime heat entry.')
        if win > 0.3: tips.append('Large glazing adds solar heat in a hot climate; keep the window ratio below about 0.25 or add shading.')
        if pcm < 0.01: tips.append('A thin PCM layer near 20 °C can flatten the day-night swing for a modest cost.')
    else:
        if not S['passive_ok']: tips.append('Passive comfort not reached: increase insulation, add a PCM layer, or raise the south window ratio to capture more winter sun.')
        if ori > 30: tips.append('Rotate the long side towards south (orientation below 30°) to raise solar gain.')
        if ins < 0.04: tips.append('Insulation is thin for a cold site; even 5-8 cm noticeably cuts heating fuel.')
        if SS['ncr'] < S['ncr'] - 0.2: tips.append('Comfort drops sharply in the cold/cloudy stress test; add thermal mass or insulation for resilience.')
    if S['cost_lakh'] > 3: tips.append('Build cost is high; try the Optimize tab with a higher cost weight.')
    return tips or ['This design looks balanced for the selected site. Confirm with real logger data on the Calibration tab.']

def context(page, site, mat, design, wx, uploaded=None, opt=None, calibration=None):
    S = sv.summarize(design, wx, mat)
    wx_stress = sv.synthetic(site, cloud=0.7, dT=-6)
    SS = sv.summarize(design, wx_stress, mat)
    return dict(page=page, site=site, mat=mat, design=design, S=S, SS=SS,
                plots=make_plots(S, wx, SS, wx_stress), has_jax=sv.HAS_JAX, tips=advice(S, SS, design, site), zone=sv.CLIMATE.get(site.split(' - ')[0], 'Custom'), sites=sv.SITES,
                materials=sv.MATERIALS, names=sv.NAMES, labels=LABELS, lo=sv.LO, hi=sv.HI,
                opt=opt, calibration=calibration)

@app.route('/', methods=['GET', 'POST'])
def index():
    site = request.values.get('site', session.get('site', list(sv.SITES)[0]))
    if site not in sv.SITES: site = list(sv.SITES)[0]
    mat = request.values.get('mat', session.get('mat', 'mud'))
    if mat not in sv.MATERIALS: mat = 'mud'
    design = current_design()
    if request.method == 'POST':
        action = request.form.get('action', 'simulate')
        if action in ('simulate', 'optimise', 'calibrate'):
            vals = []
            for i in range(len(sv.NAMES)):
                try: vals.append(float(request.form.get(f'v{i}', design[i])))
                except (TypeError, ValueError): vals.append(float(design[i]))
            design = np.clip(np.asarray(vals), sv.LO, sv.HI)
        session['design'] = design.tolist(); session['site'] = site; session['mat'] = mat
        if action == 'add_material':
            try:
                name = request.form['material_name'].strip()
                key = ''.join(c.lower() if c.isalnum() else '_' for c in name).strip('_')
                if not key: raise ValueError('Please enter a valid material name.')
                sv.add_material(key, name, float(request.form['conductivity']), float(request.form['capacity']) * 1e6, float(request.form['price']))
                flash(f'Material “{name}” added to this running app.')
            except Exception as e: flash(f'Could not add material: {e}')
            return redirect(url_for('index', page='simulate'))
        try:
            wx = get_weather(site, request.files.get('weather_csv') if action == 'simulate' else None)
            if action == 'optimise':
                stress = request.form.get('worst') == 'on'
                scenarios = [wx, sv.synthetic(site, cloud=0.7, dT=-6)] if stress else [wx]
                result = sv.optimize(scenarios, mat, w=float(request.form.get('weight', 0.3)), x0=design)
                design = np.asarray(result['x']); session['design'] = design.tolist()
                flash('Suggested design calculated and loaded into the sliders.')
                return redirect(url_for('index', page='optimise'))
            if action == 'calibrate':
                uploaded = request.files.get('logger_csv')
                if uploaded and uploaded.filename:
                    import pandas as pd
                    obs = pd.read_csv(uploaded)['Ti'].values
                else: obs = sv.fake_logger(design, wx, mat)
                result = sv.calibrate(design, wx, mat, obs)
                return render_template_string(HTML, **context('calibrate', site, mat, design, wx, calibration=result))
            page = request.args.get('page', 'simulate')
            if page not in ('simulate', 'stress', 'optimise', 'calibrate'): page = 'simulate'
            return render_template_string(HTML, **context(page, site, mat, design, wx))
        except Exception as e:
            flash(f'Action failed: {e}')
            page = request.args.get('page', 'simulate')
            wx = sv.synthetic(site)
            return render_template_string(HTML, **context(page, site, mat, design, wx))
    page = request.args.get('page', 'simulate')
    if page not in ('simulate', 'stress', 'optimise', 'calibrate'): page = 'simulate'
    wx = sv.synthetic(site)
    return render_template_string(HTML, **context(page, site, mat, design, wx))

@app.get('/report')
def report():
    site = session.get('site', list(sv.SITES)[0]); mat = session.get('mat', 'mud'); x = current_design()
    S = sv.summarize(x, sv.synthetic(site), mat); SS = sv.summarize(x, sv.synthetic(site, cloud=0.7, dT=-6), mat)
    return Response(sv.markdown_report(site, mat, x, S, SS), mimetype='text/markdown',
                    headers={'Content-Disposition': 'attachment; filename=sunvault_report.md'})

@app.post('/reset')
def reset():
    session['design'] = DEFAULT_DESIGN
    flash('Design reset to default values.')
    return redirect(url_for('index', page='simulate'))


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=int(os.environ.get('PORT', 5000)), debug=True)
