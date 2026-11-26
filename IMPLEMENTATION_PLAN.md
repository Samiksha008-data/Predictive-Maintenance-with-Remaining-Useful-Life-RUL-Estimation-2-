# Implementation Plan for Predictive Maintenance & RUL Estimation Platform

## Overview
This repository will host an end‑to‑end solution for estimating the Remaining Useful Life (RUL) of NASA turbofan engines (C‑MAPSS FD001‑FD004). The project is split into three logical layers:

```
ml_pipeline/   ← data ingestion, preprocessing, feature engineering, model training
backend/       ← FastAPI service exposing training artefacts & live telemetry simulation
frontend/      ← React dashboard for fleet monitoring and prognostics
```

The plan outlines **architecture**, **technology stack**, **key deliverables**, and **step‑by‑step actions** required to get a runnable prototype.

---

## 1. Architecture & Tech Stack

| Layer | Language / Framework | Key Libraries | Purpose |
|------|----------------------|---------------|---------|
| **ML Pipeline** | Python 3.11 | pandas, numpy, scikit‑learn, torch (LSTM / CNN‑LSTM), xgboost, shap, joblib | Data prep, feature engineering, model training, evaluation, artefact serialization |
| **Backend** | Python (FastAPI) | fastapi, pydantic, uvicorn, websockets, aiofiles | Serve model artefacts, provide telemetry simulation streams, expose prediction API |
| **Frontend** | TypeScript (React‑Vite) | react, vite, tailwindcss, recharts/plotly.js, lucide‑react, axios, socket.io‑client | Interactive dashboard, real‑time charts, explainability UI |

### Communication
* Backend ↔ Frontend – JSON over HTTP (REST) for static calls, Server‑Sent Events (SSE) for telemetry streaming.
* Model artefacts stored under `ml_pipeline/artifacts/` and loaded by the FastAPI service at startup.

---

## 2. Detailed Work Packages

### 2.1 ML Pipeline (`ml_pipeline/`)
1. **Data Acquisition** – script to download the C‑MAPSS tarball (fallback to a small sampled CSV if network unavailable).
2. **Pre‑processing** –
   * Remove constant sensors (`s1,s5,s6,s10,s16,s18,s19`).
   * Scale operational settings per regime using `StandardScaler`.
   * Normalize remaining sensors with `MinMaxScaler`.
3. **Feature Engineering** –
   * Piecewise‑linear RUL cap (max 125 cycles).
   * Rolling window stats (mean, std, EMA, slope) for windows 5/10/20.
   * Health Index via PCA reconstruction error or auto‑encoder error.
   * Build sliding windows of length 30 for sequence models.
4. **Model Development** –
   * Baseline XGBoost regressor with custom asymmetric loss (implemented via `obj` callback).
   * PyTorch LSTM and optional 1‑D‑CNN‑LSTM hybrid.
   * Training loops with early stopping on validation asymmetric score.
5. **Evaluation & Explainability** –
   * Compute RMSE and NASA asymmetric score.
   * SHAP values for XGBoost, integrated gradients for PyTorch.
   * Persist best model, scalers, and feature metadata with `joblib` into `ml_pipeline/artifacts/`.
6. **Packaging** – `setup.py` / `requirements.txt` for reproducibility.

### 2.2 Backend (`backend/`)
1. **FastAPI App** – `main.py` with three endpoints:
   * `GET /api/engines` – returns list of simulated engines, current cycle, true RUL, health status.
   * `POST /api/predict` – accepts JSON array of shape `(30, n_features)`, returns predicted RUL, 95 % CI, asymmetric risk score, top‑3 sensor contributions.
   * `GET /api/simulate/{engine_id}` – SSE stream emitting one telemetry row per cycle.
2. **Model Loader** – lazy load artefacts at startup, share across requests.
3. **Telemetry Simulator** – generator that yields rows from the original FD001 file for a given engine ID.
4. **Dependency Management** – `requirements.txt` (fastapi, uvicorn, pydantic, numpy, pandas, torch, xgboost, shap, sse-starlette).
5. **Testing** – pytest suite covering each endpoint.

### 2.3 Frontend (`frontend/`)
1. **Project Setup** – Vite + React + TypeScript scaffold.
2. **UI Layout** – dark‑mode Tailwind theme with:
   * Fleet overview cards.
   * Engine selector + playback controls (play, pause, scrub slider).
   * Degradation charts (RUL vs. actual, sensor time‑series).
   * Explainability panel (SHAP bar chart, maintenance recommendation).
3. **Data Layer** – `api.ts` using `axios` for REST calls and `EventSource` for SSE telemetry.
4. **Visualization** – Recharts (or Plotly) for line charts, confidence bands, and gauge widgets.
5. **State Management** – React Context or Zustand for selected engine, telemetry buffer, and prediction results.
6. **Build & Deploy** – `npm install`, `npm run build`, and `npm run dev` scripts.

---

## 3. Milestone Timeline (approx.)
| Milestone | Description | Approx. Duration |
|-----------|-------------|------------------|
| **M1 – Scaffold** | Create directory layout, write IMPLEMENTATION_PLAN, add README placeholders. | 0.5 day |
| **M2 – Data Pipeline** | Implement download, cleaning, feature engineering, baseline XGBoost. | 2 days |
| **M3 – Deep Model** | Build LSTM/CNN‑LSTM, train, evaluate, store artefacts. | 3 days |
| **M4 – Backend API** | FastAPI server, model loading, telemetry SSE, unit tests. | 2 days |
| **M5 – Frontend UI** | React dashboard, API integration, live replay, explainability UI. | 3 days |
| **M6 – End‑to‑End Validation** | Run end‑to‑end demo, fine‑tune thresholds, write documentation. | 1 day |

---

## 4. Commands to Run the Solution (once code is in place)
```bash
# 1. Set up virtual environment (Python 3.11)
python -m venv .venv && .\.venv\Scripts\activate
pip install -r ml_pipeline/requirements.txt

# 2. Train models (produces artifacts/ folder)
python -m ml_pipeline.train

# 3. Start FastAPI backend (runs on http://127.0.0.1:8000)
uvicorn backend.main:app --reload

# 4. Frontend – from the frontend directory
cd frontend
npm install
npm run dev   # dev server at http://localhost:5173
```

---

## 5. Next Steps
* Populate each sub‑directory with starter code (`ml_pipeline/train.py`, `backend/main.py`, `frontend/src/*`).
* Run the training script to generate artefacts.
* Verify the FastAPI endpoints with `curl` or the provided pytest suite.
* Launch the React dev server and connect it to the backend.

Feel free to request any of the concrete code files or further refinements.
