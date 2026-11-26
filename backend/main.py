"""
backend/main.py
----------------
FastAPI service for NASA C-MAPSS Predictive Maintenance & RUL Estimation.

Endpoints:
* GET  /api/health                  - service & model readiness check
* GET  /api/overview                - fleet KPIs & evaluation metrics
* GET  /api/engines                 - list all test engines with cycle, predicted/true RUL, and health status
* GET  /api/engines/{engine_id}     - full cycle-by-cycle trajectory for scrubber & offline inspection
* POST /api/predict                 - predict RUL, 95% CI, NASA asymmetric risk, and top-3 SHAP sensors
* GET  /api/simulate/{engine_id}    - SSE stream of cycle-by-cycle telemetry from cycle 1 to end of record
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import shap
import torch
import xgboost as xgb
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ml_pipeline.data_loader import load_data
from ml_pipeline.features import (
    add_pca_health_index,
    compute_rolling_features,
    drop_constant_sensors,
    piecewise_rul,
    scale_operating_settings,
    scale_sensors,
)
from ml_pipeline.metrics import nasa_asymmetric_score
from ml_pipeline.train import LSTMRegressor

ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT_DIR / "ml_pipeline" / "artifacts"

SENSOR_METADATA: Dict[str, Dict[str, str]] = {
    "s2": {"name": "LPC Outlet Temp", "unit": "°R"},
    "s3": {"name": "HPC Outlet Temp", "unit": "°R"},
    "s4": {"name": "LPT Outlet Temp", "unit": "°R"},
    "s7": {"name": "HPC Outlet Pressure", "unit": "psia"},
    "s8": {"name": "Physical Fan Speed", "unit": "rpm"},
    "s9": {"name": "Physical Core Speed", "unit": "rpm"},
    "s11": {"name": "HPC Static Pressure", "unit": "psia"},
    "s12": {"name": "Fuel Flow Ratio", "unit": "pps/psi"},
    "s13": {"name": "Corrected Fan Speed", "unit": "rpm"},
    "s14": {"name": "Corrected Core Speed", "unit": "rpm"},
    "s15": {"name": "Bypass Ratio", "unit": "ratio"},
    "s17": {"name": "Bleed Enthalpy", "unit": "BTU/lbm"},
    "s20": {"name": "HPT Coolant Bleed", "unit": "lbm/s"},
    "s21": {"name": "LPT Coolant Bleed", "unit": "lbm/s"},
}


def classify_health(rul: float) -> str:
    if rul < 25.0:
        return "Critical"
    if rul < 60.0:
        return "Warning"
    return "Healthy"


def maintenance_recommendation(rul: float) -> str:
    if rul < 25.0:
        return "IMMEDIATE GROUNDING: Schedule borescope inspection & hot-section overhaul within 10 cycles."
    if rul < 60.0:
        return "MAINTENANCE WINDOW: Plan predictive line maintenance & compressor wash within 20–30 cycles."
    return "NOMINAL OPERATION: Continue standard flight-cycle monitoring; next scheduled check in 60+ cycles."


class AppState:
    def __init__(self) -> None:
        self.loaded: bool = False
        self.metadata: Dict[str, Any] = {}
        self.feature_names: List[str] = []
        self.transformers: Dict[str, Any] = {}
        self.xgb_model: Optional[xgb.Booster] = None
        self.lstm_model: Optional[LSTMRegressor] = None
        self.shap_explainer: Optional[shap.TreeExplainer] = None
        self.engines_summary: List[Dict[str, Any]] = []
        self.engine_trajectories: Dict[int, List[Dict[str, Any]]] = {}
        self.engine_feature_matrices: Dict[int, np.ndarray] = {}
        self.fleet_overview: Dict[str, Any] = {}

    def load(self) -> None:
        if self.loaded:
            return

        if not (ARTIFACTS_DIR / "metadata.json").exists():
            raise RuntimeError(
                f"Model artifacts not found in {ARTIFACTS_DIR}. Run `python -m ml_pipeline.train` first."
            )

        with open(ARTIFACTS_DIR / "metadata.json", "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.feature_names = self.metadata["features"]
        self.transformers = joblib.load(ARTIFACTS_DIR / "preprocess.pkl")

        self.xgb_model = xgb.Booster()
        self.xgb_model.load_model(str(ARTIFACTS_DIR / "xgb_best.json"))
        self.shap_explainer = shap.TreeExplainer(self.xgb_model)

        ckpt = torch.load(ARTIFACTS_DIR / "lstm_best.pt", map_location="cpu", weights_only=True)
        self.lstm_model = LSTMRegressor(input_dim=int(ckpt["input_dim"]))
        self.lstm_model.load_state_dict(ckpt["state_dict"])
        self.lstm_model.eval()

        self._build_test_engine_cache()
        self.loaded = True

    def _compute_shap_top_sensors(self, flat_row: np.ndarray, top_k: int = 3) -> List[Dict[str, Any]]:
        assert self.shap_explainer is not None
        shap_vals = self.shap_explainer.shap_values(flat_row.reshape(1, -1))[0]

        sensor_scores: Dict[str, float] = {}
        sensor_signed: Dict[str, float] = {}
        for feat_name, sval in zip(self.feature_names, shap_vals):
            if not feat_name.startswith("s"):
                continue
            base_sensor = feat_name.split("_")[0]
            sensor_scores[base_sensor] = sensor_scores.get(base_sensor, 0.0) + float(abs(sval))
            sensor_signed[base_sensor] = sensor_signed.get(base_sensor, 0.0) + float(sval)

        total = sum(sensor_scores.values()) or 1.0
        ranked = sorted(sensor_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for sensor, mag in ranked:
            meta = SENSOR_METADATA.get(sensor, {"name": sensor.upper(), "unit": ""})
            signed = sensor_signed[sensor]
            results.append(
                {
                    "sensor": sensor,
                    "label": f"{meta['name']} ({sensor})",
                    "importance_pct": round((mag / total) * 100.0, 2),
                    "shap_impact": round(signed, 3),
                    "direction": "degrading" if signed < 0 else "stabilizing",
                }
            )
        return results

    def predict_window(
        self, window: np.ndarray, true_rul: Optional[float] = None
    ) -> Dict[str, Any]:
        assert self.xgb_model is not None and self.lstm_model is not None

        if window.ndim != 2 or window.shape[0] != 30 or window.shape[1] != len(self.feature_names):
            raise HTTPException(
                status_code=422,
                detail=f"Expected window shape (30, {len(self.feature_names)}), got {tuple(window.shape)}",
            )

        flat_row = window[-1, :].astype(np.float32)
        xgb_pred = float(np.clip(self.xgb_model.predict(xgb.DMatrix(flat_row.reshape(1, -1)))[0], 0.0, 125.0))

        with torch.no_grad():
            seq_tensor = torch.from_numpy(window.astype(np.float32)).unsqueeze(0)
            lstm_pred = float(np.clip(self.lstm_model(seq_tensor).item(), 0.0, 125.0))

        best_model = self.metadata.get("best_model", "xgboost")
        primary_pred = xgb_pred if best_model == "xgboost" else lstm_pred

        # 95% confidence interval combining test RMSE and model disagreement
        base_rmse = float(self.metadata.get("test_rmse", 15.0))
        spread = max(base_rmse * 0.55, abs(xgb_pred - lstm_pred) * 1.25 + 4.5)
        ci_low = round(max(0.0, primary_pred - 1.96 * (spread / 2.0)), 2)
        ci_high = round(min(125.0, primary_pred + 1.96 * (spread / 2.0)), 2)

        if true_rul is not None:
            asym_risk = round(
                nasa_asymmetric_score(np.array([true_rul]), np.array([primary_pred])), 3
            )
        else:
            # Operational risk proxy increases steeply as RUL approaches catastrophic threshold
            asym_risk = round(float(np.exp(max(0.0, 35.0 - primary_pred) / 10.0) - 1.0), 3)

        top_sensors = self._compute_shap_top_sensors(flat_row, top_k=3)
        status = classify_health(primary_pred)

        return {
            "predicted_rul": round(primary_pred, 2),
            "xgb_rul": round(xgb_pred, 2),
            "lstm_rul": round(lstm_pred, 2),
            "confidence_interval": [ci_low, ci_high],
            "asymmetric_risk_score": asym_risk,
            "health_status": status,
            "top_contributing_sensors": top_sensors,
            "maintenance_recommendation": maintenance_recommendation(primary_pred),
        }

    def _build_test_engine_cache(self) -> None:
        _, test_df, rul_series = load_data("FD001")
        test_units = sorted(test_df["unit"].unique())
        rul_map = dict(zip(test_units, rul_series.to_numpy()))
        test_max_cycle = test_df.groupby("unit")["cycle"].transform("max")
        test_end_rul = test_df["unit"].map(rul_map)
        test_df["true_rul_raw"] = test_end_rul + (test_max_cycle - test_df["cycle"])
        test_df["true_rul"] = piecewise_rul(test_df["true_rul_raw"], max_rul=125)

        proc_df = drop_constant_sensors(test_df)
        proc_df, _ = scale_operating_settings(proc_df, self.transformers["op_scaler"])
        proc_df, _ = scale_sensors(proc_df, self.transformers["sensor_scaler"])
        proc_df = compute_rolling_features(proc_df)
        proc_df, _ = add_pca_health_index(proc_df, pca=self.transformers["pca"])

        # Normalize PCA reconstruction error into a 0-100 Health Index (100 = healthy, 0 = degraded)
        hi_raw = proc_df["health_index"].to_numpy()
        hi_p99 = float(np.percentile(hi_raw, 99)) or 1.0
        proc_df["health_score"] = np.clip(100.0 * (1.0 - hi_raw / hi_p99), 5.0, 100.0)

        # Batch predictions across all valid 30-cycle windows for fast startup
        window_size = int(self.metadata.get("window_size", 30))
        all_windows: List[np.ndarray] = []
        window_keys: List[tuple[int, int]] = []

        for unit, group in proc_df.groupby("unit"):
            group = group.sort_values("cycle")
            feat_mat = group[self.feature_names].to_numpy(dtype=np.float32)
            cyc_arr = group["cycle"].to_numpy(dtype=int)
            for i in range(len(group) - window_size + 1):
                all_windows.append(feat_mat[i : i + window_size])
                window_keys.append((int(unit), int(cyc_arr[i + window_size - 1])))

        win_arr = np.asarray(all_windows, dtype=np.float32)
        xgb_preds = np.clip(
            self.xgb_model.predict(xgb.DMatrix(win_arr[:, -1, :])), 0.0, 125.0
        )
        with torch.no_grad():
            lstm_preds = np.clip(
                self.lstm_model(torch.from_numpy(win_arr)).numpy(), 0.0, 125.0
            )

        best_model = self.metadata.get("best_model", "xgboost")
        base_rmse = float(self.metadata.get("test_rmse", 15.0))
        pred_lookup: Dict[tuple[int, int], tuple[float, float, float]] = {}
        for idx, key in enumerate(window_keys):
            xgb_v = float(xgb_preds[idx])
            lstm_v = float(lstm_preds[idx])
            primary_v = xgb_v if best_model == "xgboost" else lstm_v
            pred_lookup[key] = (primary_v, xgb_v, lstm_v)

        key_sensors = ["s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12", "s14", "s15", "s17", "s20", "s21"]
        self.engines_summary = []
        self.engine_trajectories = {}

        for unit, group in proc_df.groupby("unit"):
            unit_id = int(unit)
            group = group.sort_values("cycle")
            traj: List[Dict[str, Any]] = []

            for _, row in group.iterrows():
                cyc = int(row["cycle"])
                true_rul = float(row["true_rul"])
                if (unit_id, cyc) in pred_lookup:
                    pred_rul, xgb_v, lstm_v = pred_lookup[(unit_id, cyc)]
                else:
                    # Warm-up cycles (< 30) use initial steady-state estimate capped at 125
                    pred_rul = 125.0
                    xgb_v = 125.0
                    lstm_v = 125.0

                spread = max(base_rmse * 0.55, abs(xgb_v - lstm_v) * 1.25 + 4.5)
                ci_low = round(max(0.0, pred_rul - 1.96 * (spread / 2.0)), 2)
                ci_high = round(min(125.0, pred_rul + 1.96 * (spread / 2.0)), 2)
                asym = round(nasa_asymmetric_score(np.array([true_rul]), np.array([pred_rul])), 3)

                traj.append(
                    {
                        "engine_id": unit_id,
                        "cycle": cyc,
                        "true_rul": round(true_rul, 2),
                        "predicted_rul": round(pred_rul, 2),
                        "xgb_rul": round(xgb_v, 2),
                        "lstm_rul": round(lstm_v, 2),
                        "confidence_interval": [ci_low, ci_high],
                        "asymmetric_risk_score": asym,
                        "health_index": round(float(row["health_score"]), 2),
                        "health_status": classify_health(pred_rul),
                        "sensors": {s: round(float(row[s]), 4) for s in key_sensors if s in row},
                    }
                )

            # Precompute SHAP top-3 sensors for the final cycle of each engine
            feat_matrix = group[self.feature_names].to_numpy(dtype=np.float32)
            self.engine_feature_matrices[unit_id] = feat_matrix
            last_feat = feat_matrix[-1]
            top_sensors = self._compute_shap_top_sensors(last_feat, top_k=3)
            last_point = traj[-1]
            last_point["top_contributing_sensors"] = top_sensors
            last_point["maintenance_recommendation"] = maintenance_recommendation(last_point["predicted_rul"])

            self.engine_trajectories[unit_id] = traj
            self.engines_summary.append(
                {
                    "engine_id": unit_id,
                    "current_cycle": last_point["cycle"],
                    "predicted_rul": last_point["predicted_rul"],
                    "true_rul": last_point["true_rul"],
                    "health_index": last_point["health_index"],
                    "health_status": last_point["health_status"],
                    "asymmetric_risk_score": last_point["asymmetric_risk_score"],
                }
            )

        critical_count = sum(1 for e in self.engines_summary if e["health_status"] == "Critical")
        warning_count = sum(1 for e in self.engines_summary if e["health_status"] == "Warning")
        healthy_count = sum(1 for e in self.engines_summary if e["health_status"] == "Healthy")
        avg_hi = round(
            float(np.mean([e["health_index"] for e in self.engines_summary])), 2
        )
        self.fleet_overview = {
            "total_engines": len(self.engines_summary),
            "critical_alerts": critical_count,
            "warning_alerts": warning_count,
            "healthy_engines": healthy_count,
            "avg_health_index": avg_hi,
            "best_model": self.metadata.get("best_model", "xgboost"),
            "model_rmse": round(float(self.metadata.get("test_rmse", 0.0)), 3),
            "model_asymmetric_score": round(float(self.metadata.get("test_asymmetric_score", 0.0)), 3),
            "test_last_cycle_metrics": self.metadata.get("test_last_cycle_metrics", {}),
            "feature_count": len(self.feature_names),
        }


state = AppState()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    state.load()
    yield


app = FastAPI(
    title="NASA C-MAPSS Predictive Maintenance API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    window: List[List[float]] = Field(..., description="Sliding window of shape (30, 186)")
    true_rul: Optional[float] = Field(default=None, description="Optional true RUL for asymmetric score evaluation")


@app.get("/api/health")
async def health():
    state.load()
    return {
        "status": "ok",
        "best_model": state.metadata.get("best_model"),
        "feature_count": len(state.feature_names),
        "engines_loaded": len(state.engines_summary),
    }


@app.get("/api/overview")
async def get_overview():
    state.load()
    return state.fleet_overview


@app.get("/api/engines")
async def list_engines():
    state.load()
    return state.engines_summary


@app.get("/api/engines/{engine_id}")
async def get_engine_detail(engine_id: int):
    state.load()
    if engine_id not in state.engine_trajectories:
        raise HTTPException(status_code=404, detail=f"Engine #{engine_id} not found")
    traj = state.engine_trajectories[engine_id]
    last = traj[-1]
    return {
        "engine_id": engine_id,
        "total_cycles": len(traj),
        "latest": last,
        "trajectory": traj,
    }


@app.get("/api/engines/{engine_id}/explain/{cycle}")
async def explain_engine_cycle(engine_id: int, cycle: int):
    state.load()
    if engine_id not in state.engine_trajectories:
        raise HTTPException(status_code=404, detail=f"Engine #{engine_id} not found")
    traj = state.engine_trajectories[engine_id]
    if cycle < 1 or cycle > len(traj):
        raise HTTPException(status_code=400, detail=f"Cycle must be between 1 and {len(traj)}")
    feat_row = state.engine_feature_matrices[engine_id][cycle - 1]
    point = traj[cycle - 1]
    return {
        "engine_id": engine_id,
        "cycle": cycle,
        "predicted_rul": point["predicted_rul"],
        "health_status": point["health_status"],
        "top_contributing_sensors": state._compute_shap_top_sensors(feat_row, top_k=3),
        "maintenance_recommendation": maintenance_recommendation(point["predicted_rul"]),
    }


@app.post("/api/predict")
async def predict(req: PredictRequest):
    state.load()
    arr = np.asarray(req.window, dtype=np.float32)
    return state.predict_window(arr, true_rul=req.true_rul)


@app.get("/api/simulate/{engine_id}")
async def simulate(
    engine_id: int,
    delay: float = Query(default=0.08, ge=0.0, le=2.0, description="Seconds between streamed cycles"),
    start_cycle: int = Query(default=1, ge=1),
):
    state.load()
    if engine_id not in state.engine_trajectories:
        raise HTTPException(status_code=404, detail=f"Engine #{engine_id} not found")

    traj = state.engine_trajectories[engine_id]
    feat_matrix = state.engine_feature_matrices[engine_id]

    async def event_stream():
        for idx, point in enumerate(traj):
            if point["cycle"] < start_cycle:
                continue
            payload = dict(point)
            payload["top_contributing_sensors"] = state._compute_shap_top_sensors(feat_matrix[idx], top_k=3)
            payload["maintenance_recommendation"] = maintenance_recommendation(point["predicted_rul"])
            payload["done"] = point["cycle"] == traj[-1]["cycle"]
            yield f"data: {json.dumps(payload)}\n\n"
            if delay > 0:
                await asyncio.sleep(delay)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
