# 🚦 TrafficSense AI

> **ML-powered traffic volume prediction** — enter time, date, and weather conditions to instantly forecast how many vehicles will be on the road.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?style=flat-square&logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?style=flat-square&logo=scikitlearn)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📌 What It Does

TrafficSense AI predicts **hourly vehicle count** on a highway based on:
- 🕐 Time of day, day of week, month, year
- 🌤 Weather conditions (temperature, rain, snow, cloud cover)
- 📅 Whether it's a public holiday or weekend

The model instantly classifies traffic as **Low / Moderate / High** and shows a confidence-backed result with a visual load indicator.

---

## 🧠 Model Details

| Property | Value |
|---|---|
| Algorithm | Random Forest Regressor |
| Training samples | 48,204 rows |
| R² Score | **0.942** |
| MAE | ±312 vehicles/hr |
| Trees | 100 |
| Dataset | Metro Interstate Traffic Volume (UCI) |

---

## 🗂 Project Structure

```
trafficsense-ai/
│
├── app.py                  # Streamlit frontend
├── requirements.txt        # Python dependencies
├── README.md               # This file
│
├── models/
│   ├── model.pkl           # Trained Random Forest model
│   ├── scaler.pkl          # StandardScaler for weather features
│   └── feature_columns.pkl # Ordered feature list
│
└── notebooks/              # (optional) training notebooks
```

---

## 🚀 Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/trafficsense-ai.git
cd trafficsense-ai
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## ☁️ Live Demo

🔗 **https://trafficsenseai-keia85kbh2rgphjpmuaeap.streamlit.app/** 

---

## 📊 How to Use

1. **Set the time** — choose hour, month, year and day of week
2. **Set weather** — adjust temperature, rain, snow, cloud cover and pick a weather type
3. **Hit "Run Prediction"** — results appear instantly below
4. **Read the result** — see predicted vehicles/hr, traffic level badge, and load gauge

---

## 🛠 Tech Stack

- **Frontend** — Streamlit with custom CSS (DM Sans + Sora fonts, light theme)
- **ML Model** — scikit-learn Random Forest Regressor
- **Data processing** — pandas, numpy
- **Model persistence** — joblib
- **Deployment** — Streamlit Community Cloud (free)

---

## 📈 Dataset

[Metro Interstate Traffic Volume](https://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume) from the UCI Machine Learning Repository.

Features used: hourly weather data + date/time features → target: `traffic_volume`

---

## 👤 Author
Ritwika Dey
Made as a Mini Project · Streamlit + scikit-learn
