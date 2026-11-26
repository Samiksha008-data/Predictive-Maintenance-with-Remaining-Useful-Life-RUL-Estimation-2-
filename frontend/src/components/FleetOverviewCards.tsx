import { Activity, AlertTriangle, Cpu, ShieldCheck } from "lucide-react";
import { OverviewMetrics } from "../types";

interface Props {
  overview: OverviewMetrics | null;
}

export function FleetOverviewCards({ overview }: Props) {
  const cards = [
    {
      title: "Total Active Engines",
      value: overview ? `${overview.total_engines}` : "—",
      sub: overview
        ? `${overview.healthy_engines} Healthy · ${overview.warning_alerts} Warning`
        : "Loading C-MAPSS FD001 fleet...",
      icon: Cpu,
      accent: "text-sky-400",
      border: "border-slate-800",
      bg: "bg-sky-500/10",
    },
    {
      title: "Critical Alerts (RUL < 25)",
      value: overview ? `${overview.critical_alerts}` : "—",
      sub: "Immediate borescope / grounding required",
      icon: AlertTriangle,
      accent: "text-rose-400",
      border: overview && overview.critical_alerts > 0 ? "border-rose-500/40" : "border-slate-800",
      bg: "bg-rose-500/10",
    },
    {
      title: "Avg Fleet Health Index",
      value: overview ? `${overview.avg_health_index.toFixed(1)}%` : "—",
      sub: "Composite PCA reconstruction score",
      icon: ShieldCheck,
      accent: "text-emerald-400",
      border: "border-slate-800",
      bg: "bg-emerald-500/10",
    },
    {
      title: "Model Asymmetric Score",
      value: overview ? `${overview.model_asymmetric_score.toFixed(1)}` : "—",
      sub: overview
        ? `${overview.best_model.toUpperCase()} · Test RMSE ${overview.model_rmse.toFixed(2)} cyc`
        : "Evaluating NASA S-score...",
      icon: Activity,
      accent: "text-amber-400",
      border: "border-slate-800",
      bg: "bg-amber-500/10",
    },
  ];

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
      {cards.map((card) => {
        const Icon = card.icon;
        return (
          <div
            key={card.title}
            className={`rounded-xl border ${card.border} bg-slate-900/80 p-4 shadow-lg backdrop-blur`}
          >
            <div className="flex items-start justify-between">
              <div>
                <p className="text-xs font-medium uppercase tracking-wider text-slate-400">
                  {card.title}
                </p>
                <p className={`mt-2 font-mono text-3xl font-bold ${card.accent}`}>
                  {card.value}
                </p>
              </div>
              <div className={`rounded-lg ${card.bg} p-2.5`}>
                <Icon className={`h-5 w-5 ${card.accent}`} />
              </div>
            </div>
            <p className="mt-2 text-xs text-slate-400">{card.sub}</p>
          </div>
        );
      })}
    </div>
  );
}
