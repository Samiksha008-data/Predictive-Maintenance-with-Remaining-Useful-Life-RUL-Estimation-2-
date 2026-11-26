import {
  Area,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { TelemetryPoint } from "../types";

interface Props {
  data: TelemetryPoint[];
}

interface ChartRow {
  cycle: number;
  predicted_rul: number;
  true_rul: number;
  lstm_rul: number;
  ciBase: number;
  ciSpan: number;
  ciLow: number;
  ciHigh: number;
}

interface CustomTooltipProps {
  active?: boolean;
  payload?: Array<{ payload: ChartRow }>;
  label?: number | string;
}

function DegradationTooltip({ active, payload, label }: CustomTooltipProps) {
  if (!active || !payload || payload.length === 0) return null;
  const row = payload[0].payload;
  return (
    <div className="rounded-lg border border-slate-700 bg-slate-900/95 px-3 py-2 text-xs shadow-xl">
      <p className="mb-1 font-mono font-semibold text-slate-200">Cycle {label}</p>
      <p className="text-sky-400">
        Predicted RUL: <span className="font-mono font-semibold">{row.predicted_rul.toFixed(1)} cyc</span>
      </p>
      <p className="text-emerald-400">
        Actual Piecewise RUL: <span className="font-mono font-semibold">{row.true_rul.toFixed(1)} cyc</span>
      </p>
      <p className="text-slate-400">
        95% CI: <span className="font-mono">[{row.ciLow.toFixed(1)}, {row.ciHigh.toFixed(1)}]</span>
      </p>
    </div>
  );
}

export function DegradationChart({ data }: Props) {
  const chartData: ChartRow[] = data.map((d) => ({
    cycle: d.cycle,
    predicted_rul: d.predicted_rul,
    true_rul: d.true_rul,
    lstm_rul: d.lstm_rul,
    ciBase: d.confidence_interval[0],
    ciSpan: Math.max(0, d.confidence_interval[1] - d.confidence_interval[0]),
    ciLow: d.confidence_interval[0],
    ciHigh: d.confidence_interval[1],
  }));

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/80 p-4 shadow-lg">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div>
          <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-200">
            Predicted vs. Actual Piecewise RUL
          </h2>
          <p className="text-xs text-slate-400">
            Shaded 95% confidence band with Catastrophic Failure Threshold (RUL = 25 cycles)
          </p>
        </div>
        <span className="rounded border border-rose-500/30 bg-rose-500/10 px-2 py-0.5 font-mono text-[11px] text-rose-300">
          Critical Zone &lt; 25 Cycles
        </span>
      </div>

      <div className="h-72 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={chartData} margin={{ top: 10, right: 16, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis
              dataKey="cycle"
              stroke="#64748b"
              tick={{ fill: "#94a3b8", fontSize: 11 }}
              label={{ value: "Flight Cycle", position: "insideBottom", offset: -2, fill: "#64748b", fontSize: 11 }}
            />
            <YAxis
              domain={[0, 130]}
              stroke="#64748b"
              tick={{ fill: "#94a3b8", fontSize: 11 }}
              label={{ value: "RUL (cycles)", angle: -90, position: "insideLeft", fill: "#64748b", fontSize: 11 }}
            />
            <Tooltip content={<DegradationTooltip />} />
            <Legend wrapperStyle={{ fontSize: "12px" }} />

            {/* Catastrophic Failure Threshold Zone */}
            <ReferenceArea y1={0} y2={25} fill="#ef4444" fillOpacity={0.12} />
            <ReferenceLine
              y={25}
              stroke="#f43f5e"
              strokeDasharray="4 4"
              strokeWidth={1.5}
              label={{ value: "Failure Threshold (25)", fill: "#fda4af", fontSize: 10, position: "insideTopRight" }}
            />

            {/* 95% Confidence Interval Band via stacked area */}
            <Area
              type="monotone"
              dataKey="ciBase"
              stackId="ci"
              stroke="none"
              fill="transparent"
              legendType="none"
              isAnimationActive={false}
            />
            <Area
              type="monotone"
              dataKey="ciSpan"
              stackId="ci"
              stroke="none"
              fill="#38bdf8"
              fillOpacity={0.18}
              name="95% Confidence Band"
              isAnimationActive={false}
            />

            <Line
              type="monotone"
              dataKey="true_rul"
              name="Actual Piecewise RUL"
              stroke="#10b981"
              strokeWidth={2}
              strokeDasharray="5 4"
              dot={false}
              isAnimationActive={false}
            />
            <Line
              type="monotone"
              dataKey="predicted_rul"
              name="Predicted RUL (Best Model)"
              stroke="#38bdf8"
              strokeWidth={2.5}
              dot={false}
              isAnimationActive={false}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
