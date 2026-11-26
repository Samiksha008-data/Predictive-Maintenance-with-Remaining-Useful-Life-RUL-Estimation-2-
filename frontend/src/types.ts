export type HealthStatus = "Healthy" | "Warning" | "Critical";

export interface SensorContribution {
  sensor: string;
  label: string;
  importance_pct: number;
  shap_impact: number;
  direction: "degrading" | "stabilizing";
}

export interface OverviewMetrics {
  total_engines: number;
  critical_alerts: number;
  warning_alerts: number;
  healthy_engines: number;
  avg_health_index: number;
  best_model: string;
  model_rmse: number;
  model_asymmetric_score: number;
  feature_count: number;
  test_last_cycle_metrics?: {
    xgboost?: { rmse: number; nasa_asymmetric_score: number };
    lstm?: { rmse: number; nasa_asymmetric_score: number };
  };
}

export interface EngineSummary {
  engine_id: number;
  current_cycle: number;
  predicted_rul: number;
  true_rul: number;
  health_index: number;
  health_status: HealthStatus;
  asymmetric_risk_score: number;
}

export interface TelemetryPoint {
  engine_id: number;
  cycle: number;
  true_rul: number;
  predicted_rul: number;
  xgb_rul: number;
  lstm_rul: number;
  confidence_interval: [number, number];
  asymmetric_risk_score: number;
  health_index: number;
  health_status: HealthStatus;
  sensors: Record<string, number>;
  top_contributing_sensors?: SensorContribution[];
  maintenance_recommendation?: string;
  done?: boolean;
}

export interface EngineDetailResponse {
  engine_id: number;
  total_cycles: number;
  latest: TelemetryPoint;
  trajectory: TelemetryPoint[];
}

export interface ExplainCycleResponse {
  engine_id: number;
  cycle: number;
  predicted_rul: number;
  health_status: HealthStatus;
  top_contributing_sensors: SensorContribution[];
  maintenance_recommendation: string;
}
