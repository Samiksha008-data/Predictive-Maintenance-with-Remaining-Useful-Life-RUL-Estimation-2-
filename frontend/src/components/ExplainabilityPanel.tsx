import { AlertOctagon, CheckCircle2, Gauge, Wrench } from "lucide-react";
import { ExplainCycleResponse, TelemetryPoint } from "../types";

interface Props {
  currentPoint: TelemetryPoint | null;
  explanation: ExplainCycleResponse | null;
}

export function ExplainabilityPanel({ currentPoint, explanation }: Props) {
  const status = explanation?.health_status ?? currentPoint?.health_status ?? "Healthy";
  const predRul = currentPoint?.predicted_rul ?? 0;
  const trueRul = currentPoint?.true_rul ?? 0;
  const ci = currentPoint?.confidence_interval ?? [0, 0];
  const asymRisk = currentPoint?.asymmetric_risk_score ?? 0;
  const sensors =
    explanation?.top_contributing_sensors ??
    currentPoint?.top_contributing_sensors ??
    [];
  const recommendation =
    explanation?.maintenance_recommendation ??
    currentPoint?.maintenance_recommendation ??
    "Select an engine or scrub the timeline to view maintenance scheduling recommendations.";

  const statusStyles = {
    Healthy: {
      border: "border-emerald-500/30",
      bg: "bg-emerald-500/10",
      text: "text-emerald-300",
      Icon: CheckCircle2,
    },
    Warning: {
      border: "border-amber-500/30",
      bg: "bg-amber-500/10",
      text: "text-amber-300",
      Icon: Gauge,
    },
    Critical: {
      border: "border-rose-500/40",
      bg: "bg-rose-500/15",
      text: "text-rose-300",
      Icon: AlertOctagon,
    },
  }[status];

  const StatusIcon = statusStyles.Icon;

  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
      {/* Live RUL & Risk Card */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/80 p-4 shadow-lg">
        <div className="flex items-center justify-between">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
            Prognostics Readout (Cycle {currentPoint?.cycle ?? "—"})
          </h3>
          <span
            className={`inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-xs font-semibold ${statusStyles.border} ${statusStyles.bg} ${statusStyles.text}`}
          >
            <StatusIcon className="h-3.5 w-3.5" />
            {status}
          </span>
        </div>

        <div className="mt-4 grid grid-cols-2 gap-3">
          <div className="rounded-lg border border-slate-800 bg-slate-950 p-3">
            <p className="text-[11px] uppercase text-slate-400">Predicted RUL</p>
            <p className="mt-1 font-mono text-2xl font-bold text-sky-400">
              {predRul.toFixed(1)} <span className="text-xs font-normal text-slate-400">cyc</span>
            </p>
            <p className="mt-1 font-mono text-[11px] text-slate-500">
              95% CI: [{ci[0].toFixed(1)}, {ci[1].toFixed(1)}]
            </p>
          </div>

          <div className="rounded-lg border border-slate-800 bg-slate-950 p-3">
            <p className="text-[11px] uppercase text-slate-400">True Piecewise RUL</p>
            <p className="mt-1 font-mono text-2xl font-bold text-emerald-400">
              {trueRul.toFixed(1)} <span className="text-xs font-normal text-slate-400">cyc</span>
            </p>
            <p className="mt-1 font-mono text-[11px] text-slate-500">
              Asym Risk: {asymRisk.toFixed(2)}
            </p>
          </div>
        </div>

        <div className="mt-3 flex items-center justify-between rounded-lg border border-slate-800/80 bg-slate-950/60 px-3 py-2 text-xs">
          <span className="text-slate-400">XGBoost / LSTM Comparison</span>
          <span className="font-mono text-slate-200">
            {currentPoint ? `${currentPoint.xgb_rul.toFixed(1)} / ${currentPoint.lstm_rul.toFixed(1)} cyc` : "—"}
          </span>
        </div>
      </div>

      {/* SHAP Top 3 Sensor Anomalies */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/80 p-4 shadow-lg">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
          SHAP Top-3 Degradation Drivers
        </h3>
        <p className="mt-0.5 text-xs text-slate-500">
          Sensor anomalies contributing most to current RUL estimation
        </p>

        <div className="mt-4 space-y-3">
          {sensors.length === 0 ? (
            <p className="text-xs text-slate-500">Computing SHAP attributions...</p>
          ) : (
            sensors.map((s) => (
              <div key={s.sensor}>
                <div className="mb-1 flex items-center justify-between text-xs">
                  <span className="font-medium text-slate-200">{s.label}</span>
                  <span className="font-mono text-slate-300">
                    {s.importance_pct.toFixed(1)}%{" "}
                    <span
                      className={
                        s.direction === "degrading" ? "text-rose-400" : "text-emerald-400"
                      }
                    >
                      ({s.direction})
                    </span>
                  </span>
                </div>
                <div className="h-2 w-full overflow-hidden rounded-full bg-slate-800">
                  <div
                    className={`h-full rounded-full transition-all ${
                      s.direction === "degrading" ? "bg-rose-500" : "bg-sky-500"
                    }`}
                    style={{ width: `${Math.min(100, Math.max(8, s.importance_pct))}%` }}
                  />
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      {/* Maintenance Action Panel */}
      <div className={`rounded-xl border ${statusStyles.border} bg-slate-900/80 p-4 shadow-lg`}>
        <div className="flex items-center gap-2">
          <div className={`rounded-lg ${statusStyles.bg} p-2`}>
            <Wrench className={`h-4 w-4 ${statusStyles.text}`} />
          </div>
          <div>
            <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
              Maintenance Action Window
            </h3>
            <p className="text-[11px] text-slate-400">
              Automated prognostics dispatch recommendation
            </p>
          </div>
        </div>

        <div className={`mt-4 rounded-lg border ${statusStyles.border} ${statusStyles.bg} p-3`}>
          <p className={`text-sm font-medium leading-relaxed ${statusStyles.text}`}>
            {recommendation}
          </p>
        </div>

        <div className="mt-3 flex items-center justify-between text-xs text-slate-400">
          <span>Engine Health Index</span>
          <span className="font-mono font-semibold text-slate-200">
            {currentPoint ? `${currentPoint.health_index.toFixed(1)}%` : "—"}
          </span>
        </div>
      </div>
    </div>
  );
}
