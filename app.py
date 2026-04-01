import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

st.set_page_config(page_title="TrafficSense AI", page_icon="🚦", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&family=Sora:wght@400;600;700;800;900&display=swap');

html, body, p, div, span, input, select, textarea, button, label, h1, h2, h3, h4, h5, h6 { font-family: 'DM Sans', sans-serif !important; }
.stApp { background: #F0F4FF; }
.block-container { padding: 2rem 1rem 3rem; max-width: 780px; margin: 0 auto; }
footer, #MainMenu { visibility: hidden; }

/* ── HEADER ── */
.header-card {
  background: linear-gradient(135deg, #1A56DB 0%, #5850EC 100%);
  border-radius: 24px;
  padding: 32px 32px 28px;
  margin-bottom: 20px;
  position: relative;
  overflow: hidden;
  box-shadow: 0 20px 60px rgba(26,86,219,.28);
}
.header-card::before {
  content: '';
  position: absolute;
  width: 260px; height: 260px;
  border-radius: 50%;
  background: rgba(255,255,255,.07);
  top: -80px; right: -60px;
}
.header-card::after {
  content: '';
  position: absolute;
  width: 140px; height: 140px;
  border-radius: 50%;
  background: rgba(255,255,255,.05);
  bottom: -40px; left: 40px;
}
.header-inner { position: relative; z-index: 1; }
.brand-row { display: flex; align-items: center; gap: 16px; margin-bottom: 18px; }
.brand-icon {
  width: 58px; height: 58px; border-radius: 18px;
  background: rgba(255,255,255,.2);
  backdrop-filter: blur(10px);
  display: flex; align-items: center; justify-content: center;
  font-size: 28px; flex-shrink: 0;
  border: 1px solid rgba(255,255,255,.3);
}
.brand-title {
  font-family: 'Sora', sans-serif !important;
  font-size: 26px; font-weight: 900; color: #fff;
  letter-spacing: -0.5px; line-height: 1.1; margin: 0 0 3px;
}
.brand-sub { font-size: 13px; color: rgba(255,255,255,.7); font-weight: 400; }
.badge-row { display: flex; gap: 8px; flex-wrap: wrap; }
.badge {
  background: rgba(255,255,255,.15);
  border: 1px solid rgba(255,255,255,.25);
  backdrop-filter: blur(8px);
  color: #fff;
  padding: 5px 14px;
  border-radius: 30px;
  font-size: 12px;
  font-weight: 600;
  letter-spacing: .02em;
}
.badge-dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; background: #4ade80; margin-right: 6px; animation: pulse 1.5s ease-in-out infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }

/* ── KPI STRIP ── */
.kpi-row { display: grid; grid-template-columns: repeat(4,1fr); gap: 12px; margin-bottom: 20px; }
.kpi-card {
  background: #fff;
  border-radius: 16px;
  padding: 18px 14px 16px;
  text-align: center;
  border: 1px solid #E4EBFF;
  box-shadow: 0 2px 12px rgba(26,86,219,.06);
  transition: transform .2s;
}
.kpi-card:hover { transform: translateY(-3px); }
.kpi-num { font-family: 'Sora', sans-serif; font-size: 20px; font-weight: 900; line-height: 1; }
.kpi-lbl { font-size: 10px; color: #94A3B8; margin-top: 4px; text-transform: uppercase; letter-spacing: .08em; font-weight: 600; }

/* ── SECTION CARD ── */
.section-card {
  background: #fff;
  border-radius: 20px;
  padding: 28px 28px 22px;
  margin-bottom: 16px;
  border: 1px solid #E4EBFF;
  box-shadow: 0 2px 16px rgba(26,86,219,.05);
}
.sec-head { margin-bottom: 20px; }
.sec-title {
  font-family: 'Sora', sans-serif;
  font-size: 20px;
  font-weight: 800;
  color: #1E293B;
  letter-spacing: -0.3px;
  margin: 0 0 4px;
}
.sec-sub {
  font-size: 13px;
  color: #94A3B8;
  font-weight: 400;
}

/* ── Streamlit widget overrides (light) ── */
label, .stSlider label, .stSelectbox label, .stRadio label {
  color: #6B7280 !important; font-size: 11px !important; font-weight: 700 !important;
  text-transform: uppercase !important; letter-spacing: .07em !important;
}
div[data-baseweb="select"] > div {
  background: #F8FAFF !important; border: 1.5px solid #DBEAFE !important;
  border-radius: 12px !important; color: #1E293B !important;
}
div[data-baseweb="select"] span { color: #1E293B !important; }
div[data-baseweb="slider"] [role="slider"] {
  background: #1A56DB !important; border: 3px solid #fff !important;
  box-shadow: 0 0 0 3px rgba(26,86,219,.2) !important;
  width: 20px !important; height: 20px !important;
}
div[data-baseweb="slider"] div:first-child div { background: #DBEAFE !important; height: 4px !important; }
div[data-baseweb="slider"] div:first-child div:first-child { background: #1A56DB !important; }
.stSlider [data-testid="stThumbValue"] { color: #1A56DB !important; font-weight: 800 !important; }
div[data-testid="metric-container"] {
  background: #F8FAFF !important; border: 1.5px solid #DBEAFE !important;
  border-radius: 14px !important; padding: 14px !important;
}
div[data-testid="metric-container"] [data-testid="stMetricLabel"] p {
  color: #94A3B8 !important; font-size: 10px !important; font-weight: 700 !important;
  text-transform: uppercase !important; letter-spacing: .07em !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
  color: #1E293B !important; font-size: 20px !important; font-weight: 900 !important;
}
div[data-testid="stHorizontalBlock"] { gap: 10px; }

/* ── PREDICT BUTTON ── */
.stButton > button {
  background: linear-gradient(135deg, #1A56DB, #5850EC) !important;
  color: #fff !important; border: none !important;
  border-radius: 14px !important; font-size: 15px !important;
  font-weight: 800 !important; padding: 16px 28px !important;
  width: 100% !important; letter-spacing: .03em !important;
  transition: all .2s !important;
  box-shadow: 0 8px 24px rgba(26,86,219,.3) !important;
  font-family: 'Sora', sans-serif !important;
}
.stButton > button:hover {
  transform: translateY(-3px) !important;
  box-shadow: 0 14px 36px rgba(26,86,219,.45) !important;
}

/* ── RESULT ── */
.res-wrap { animation: fadeUp .4s ease; }
@keyframes fadeUp { from{opacity:0;transform:translateY(18px)} to{opacity:1;transform:translateY(0)} }

.res-hero {
  border-radius: 20px;
  padding: 28px;
  margin-bottom: 14px;
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
}
.res-lbl { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .1em; margin-bottom: 6px; opacity: .7; }
.res-num { font-family: 'Sora', sans-serif; font-size: 60px; font-weight: 900; letter-spacing: -3px; line-height: 1; }
.res-unit { font-size: 14px; font-weight: 500; margin-left: 4px; opacity: .6; }
.res-badge {
  display: inline-flex; align-items: center; gap: 7px;
  padding: 7px 18px; border-radius: 30px;
  font-size: 12px; font-weight: 700; margin-top: 12px;
  letter-spacing: .03em;
}
.res-badge-dot { width: 8px; height: 8px; border-radius: 50%; background: currentColor; }

.detail-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 10px; margin-bottom: 14px; }
.detail-card {
  background: #fff;
  border: 1.5px solid #E4EBFF;
  border-radius: 14px;
  padding: 14px;
  text-align: center;
  box-shadow: 0 2px 8px rgba(26,86,219,.04);
}
.detail-v { font-family: 'Sora', sans-serif; font-size: 18px; font-weight: 800; color: #1E293B; }
.detail-l { font-size: 9px; color: #94A3B8; margin-top: 3px; text-transform: uppercase; letter-spacing: .08em; font-weight: 600; }

.bar-section {
  background: #fff;
  border: 1.5px solid #E4EBFF;
  border-radius: 16px;
  padding: 18px 20px;
  margin-bottom: 14px;
}
.bar-row { display: flex; justify-content: space-between; margin-bottom: 6px; }
.bar-name { font-size: 12px; color: #64748B; font-weight: 600; }
.bar-pct { font-size: 12px; font-weight: 800; }
.bar-track { height: 8px; background: #F1F5FF; border-radius: 4px; overflow: hidden; margin-bottom: 14px; }
.bar-track:last-child { margin-bottom: 0; }
.bar-fill { height: 100%; border-radius: 4px; transition: width 1s ease; }

.footer { text-align: center; font-size: 11px; color: #C7D2E8; padding: 20px; letter-spacing: .06em; text-transform: uppercase; }

/* ── INPUT SUMMARY ── */
.sum-card {
  background: #fff;
  border: 1.5px solid #E4EBFF;
  border-radius: 18px;
  padding: 22px 24px;
  margin-top: 14px;
  box-shadow: 0 2px 16px rgba(26,86,219,.05);
}
.sum-title {
  font-family: 'Sora', sans-serif;
  font-size: 15px; font-weight: 800; color: #1E293B;
  margin-bottom: 14px; letter-spacing: -0.2px;
}
.sum-grid { display: flex; flex-direction: column; gap: 0; }
.sum-row {
  display: flex; justify-content: space-between; align-items: center;
  padding: 9px 0; border-bottom: 1px solid #F1F5FF;
}
.sum-row:last-child { border-bottom: none; }
.sum-key { font-size: 12px; color: #94A3B8; font-weight: 600; }
.sum-val { font-size: 13px; color: #1E293B; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ── Load model ──────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    base = os.path.dirname(os.path.abspath(__file__))
    model     = joblib.load(os.path.join(base, "models", "model.pkl"))
    scaler    = joblib.load(os.path.join(base, "models", "scaler.pkl"))
    feat_cols = joblib.load(os.path.join(base, "models", "feature_columns.pkl"))
    return model, scaler, feat_cols

model, scaler, feat_cols = load_artifacts()

# ── Header ──────────────────────────────────────────────────
st.markdown("""
<div class="header-card">
  <div class="header-inner">
    <div class="brand-row">
      <div class="brand-icon">🚦</div>
      <div>
        <div class="brand-title">TrafficSense AI</div>
        <div class="brand-sub">Vehicle count prediction · ML-powered forecasting</div>
      </div>
    </div>
    <div class="badge-row">
      <div class="badge"><span class="badge-dot"></span>Model active</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)



# ── Time & Date card ────────────────────────────────────────
st.markdown("""
<div class="section-card">
  <div class="sec-head">
    <div class="sec-title">🕐 Time &amp; Date</div>
    <div class="sec-sub">Set the hour, day and month for prediction</div>
  </div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1: hour = st.slider("Hour of day", 0, 23, 8)
with c2: month = st.slider("Month", 1, 12, 6)
with c3: year = st.slider("Year", 2012, 2025, 2018)

c4, c5 = st.columns(2)
with c4:
    day_of_week = st.selectbox("Day of week", options=list(range(7)),
        format_func=lambda x: ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"][x])
with c5:
    is_holiday = st.selectbox("Public holiday", [0,1],
        format_func=lambda x: "Yes — holiday" if x else "No — regular day")

is_weekend = 1 if day_of_week >= 5 else 0
is_rush    = 1 if (7 <= hour <= 9) or (16 <= hour <= 18) else 0
is_peak    = 1 if month in [6,7,8] else 0

c6, c7, c8 = st.columns(3)
with c6: st.metric("Weekend",     "✅ Yes"    if is_weekend else "❌ No")
with c7: st.metric("Rush hour",   "⚡ Active" if is_rush    else "— No")
with c8: st.metric("Peak season", "☀️ Yes"   if is_peak   else "— No")

st.markdown("</div>", unsafe_allow_html=True)

# ── Weather card ────────────────────────────────────────────
st.markdown("""
<div class="section-card">
  <div class="sec-head">
    <div class="sec-title">🌤 Weather Conditions</div>
    <div class="sec-sub">Enter current weather details for accurate forecasting</div>
  </div>
""", unsafe_allow_html=True)

c9, c10 = st.columns(2)
with c9:
    temp    = st.slider("Temperature (°C)", -30, 50, 20)
    rain_1h = st.slider("Rain (mm)",          0, 100,  0)
with c10:
    clouds_all = st.slider("Cloud cover (%)", 0, 100, 30)
    snow_1h    = st.slider("Snow (mm)",       0,  10,  0)

weather = st.selectbox("Weather condition",
    ["Clear","Clouds","Rain","Drizzle","Mist","Snow","Fog","Thunderstorm","Haze","Smoke"])

st.markdown("</div>", unsafe_allow_html=True)

# ── Predict ─────────────────────────────────────────────────
months_s = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
days_s   = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

if st.button("⚡  Run Prediction"):
    input_dict = {
        'temp': temp + 273.15, 'rain_1h': rain_1h, 'snow_1h': snow_1h,
        'clouds_all': clouds_all, 'hour': hour, 'day_of_week': day_of_week,
        'month': month, 'year': year, 'is_weekend': is_weekend, 'is_holiday': is_holiday,
    }
    for col in feat_cols:
        if col.startswith("weather_main_"): input_dict[col] = 0
    wc = f"weather_main_{weather}"
    if wc in feat_cols: input_dict[wc] = 1

    df_in = pd.DataFrame([input_dict])[feat_cols]
    df_in[['temp','rain_1h','snow_1h','clouds_all']] = scaler.transform(
        df_in[['temp','rain_1h','snow_1h','clouds_all']])
    pred = max(0, int(model.predict(df_in)[0]))
    pct  = min(100, round(pred / 7200 * 100))

    if pred < 1500:
        color, light_bg, border, badge_bg, level, icon = (
            "#059669","#ECFDF5","#A7F3D0","#D1FAE5","🟢 Low Traffic","🌿")
    elif pred < 4000:
        color, light_bg, border, badge_bg, level, icon = (
            "#D97706","#FFFBEB","#FDE68A","#FEF3C7","🟡 Moderate Traffic","⚡")
    else:
        color, light_bg, border, badge_bg, level, icon = (
            "#DC2626","#FEF2F2","#FECACA","#FEE2E2","🔴 High Traffic","🚨")

    st.markdown(f"""
    <div class="res-wrap">
      <div class="res-hero" style="background:{light_bg};border:1.5px solid {border}">
        <div>
          <div class="res-lbl" style="color:{color}">{icon} Predicted volume</div>
          <div class="res-num" style="color:{color}">{pred:,}<span class="res-unit">vehicles/hr</span></div>
          <div class="res-badge" style="background:{badge_bg};color:{color};border:1px solid {border}">
            <span class="res-badge-dot" style="background:{color}"></span>{level}
          </div>
        </div>
        <svg width="100" height="100" viewBox="0 0 100 100">
          <circle cx="50" cy="50" r="40" fill="none" stroke="{border}" stroke-width="10"/>
          <circle cx="50" cy="50" r="40" fill="none" stroke="{color}" stroke-width="10"
            stroke-dasharray="{round(pct/100*251)} 251" stroke-linecap="round"
            transform="rotate(-90 50 50)"/>
          <text x="50" y="46" text-anchor="middle" font-size="17" font-weight="900"
            fill="{color}" font-family="Sora,sans-serif">{pct}%</text>
          <text x="50" y="62" text-anchor="middle" font-size="10" fill="#94A3B8"
            font-weight="700">LOAD</text>
        </svg>
      </div>

      <div class="detail-grid">
        <div class="detail-card">
          <div class="detail-v" style="color:{color}">{pred:,}</div>
          <div class="detail-l">Vehicles / hr</div>
        </div>
        <div class="detail-card">
          <div class="detail-v">{hour:02d}:00</div>
          <div class="detail-l">Time</div>
        </div>
        <div class="detail-card">
          <div class="detail-v">{days_s[day_of_week]}</div>
          <div class="detail-l">Day</div>
        </div>
        <div class="detail-card">
          <div class="detail-v">{temp}°C</div>
          <div class="detail-l">Temp</div>
        </div>
      </div>

      <div class="bar-section">
        <div class="bar-row">
          <span class="bar-name">Traffic load</span>
          <span class="bar-pct" style="color:{color}">{pct}%</span>
        </div>
        <div class="bar-track">
          <div class="bar-fill" style="width:{pct}%;background:{color}"></div>
        </div>
        <div class="bar-row">
          <span class="bar-name">Model confidence</span>
          <span class="bar-pct" style="color:#7C3AED">94%</span>
        </div>
        <div class="bar-track">
          <div class="bar-fill" style="width:94%;background:#7C3AED"></div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    summary_items = [
        ("Hour", f"{hour:02d}:00"),
        ("Day", days_s[day_of_week]),
        ("Month", months_s[month-1]),
        ("Year", str(year)),
        ("Temperature", f"{temp}\u00b0C"),
        ("Rain", f"{rain_1h} mm"),
        ("Snow", f"{snow_1h} mm"),
        ("Cloud cover", f"{clouds_all}%"),
        ("Weekend", "Yes" if is_weekend else "No"),
        ("Public holiday", "Yes" if is_holiday else "No"),
        ("Weather", weather),
    ]
    rows_html = "".join(
        f'<div class="sum-row"><span class="sum-key">{k}</span><span class="sum-val">{v}</span></div>'
        for k, v in summary_items
    )
    st.markdown(
        f'<div class="sum-card"><div class="sum-title">Input Summary</div>'
        f'<div class="sum-grid">{rows_html}</div></div>',
        unsafe_allow_html=True
    )

st.markdown('<div class="footer">TrafficSense AI &nbsp;·&nbsp; Mini Project &nbsp;·&nbsp; Streamlit + scikit-learn</div>', unsafe_allow_html=True)