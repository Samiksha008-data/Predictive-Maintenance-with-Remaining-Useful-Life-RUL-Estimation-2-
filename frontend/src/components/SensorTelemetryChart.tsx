import { useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { TelemetryPoint } from "../types";

interface Props {
  data: TelemetryPoint[];
}

const SENSOR_SERIES = [
  { key: "s4", label: "s4 · LPT Outlet Temp", color: "#f43f5e" },
  { key: "s9", label: "s9 · Core Speed", color: "#f59e0b" },
  { key: "s11", label: "s11 · HPC Static Press", color: "#38bdf8" },
  { key: "s15", label: "s15 · Bypass Ratio", color: "#a855f7" },
  { key: "s12", label: "s12 · Fuel Flow Ratio", color: "#10b981" },
];

export function SensorTelemetryChart({ data }: Props) {
  const [activeSensors, setActiveSensors] = useState<string[]>(["s4", "s9", "s11", "s15"]);

  const toggleSensor = (key: string) => {
    setActiveSensors((prev) =>
      prev.includes(key)
        ? prev.length > 1
          ? prev.filter((k) => k !== key)
          : prev
        : [...prev, key]
    );
  };

  const chartData = data.map((d) => ({
    cycle: d.cycle,
    ...d.sensors,
  }));

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/80 p-4 shadow-lg">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div>
          <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-200">
            Multi-Sensor Degradation Telemetry
          </h2>
          <p className="text-xs text-slate-400">
            MinMax-scaled sensor trajectories across engine operating cycles
          </p>
        </div>

        <div className="flex flex-wrap gap-1.5">
          {SENSOR_SERIES.map((s) => {
            const active = activeSensors.includes(s.key);
            return (
              <button
                key={s.key}
                onClick={() => toggleSensor(s.key)}
                className={`rounded-md border px-2 py-1 font-mono text-[11px] transition ${
                  active
                    ? "border-slate-600 bg-slate-800 text-slate-100"
                    : "border-slate-800 bg-slate-950 text-slate-500 hover:text-slate-300"
                }`}
              >
                <span
                  className="mr-1.5 inline-block h-2 w-2 rounded-full"
                  style={{ backgroundColor: active ? s.color : "#475569" }}
                />
                {s.key}
              </button>
            );
          })}
        </div>
      </div>

      <div className="h-72 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 10, right: 16, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis
              dataKey="cycle"
              stroke="#64748b"
              tick={{ fill: "#94a3b8", fontSize: 11 }}
            />
            <YAxis
              domain={[0, 1]}
              stroke="#64748b"
              tick={{ fill: "#94a3b8", fontSize: 11 }}
              label={{ value: "Normalized (0-1)", angle: -90, position: "insideLeft", fill: "#64748b", fontSize: 11 }}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "#0f172a",
                borderColor: "#334155",
                borderRadius: "8px",
                fontSize: "12px",
              }}
              labelFormatter={(label) => `Cycle ${label}`}
            />
            <Legend wrapperStyle={{ fontSize: "12px" }} />
            {SENSOR_SERIES.filter((s) => activeSensors.includes(s.key)).map((s) => (
              <Line
                key={s.key}
                type="monotone"
                dataKey={s.key}
                name={s.label}
                stroke={s.color}
                strokeWidth={1.8}
                dot={false}
                isAnimationActive={false}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
