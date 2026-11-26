import { Pause, Play, RotateCcw, Radio } from "lucide-react";
import { EngineSummary, HealthStatus } from "../types";

interface Props {
  engines: EngineSummary[];
  selectedEngineId: number;
  onSelectEngine: (id: number) => void;
  currentCycle: number;
  maxCycle: number;
  onScrubCycle: (cycle: number) => void;
  isPlaying: boolean;
  onTogglePlay: () => void;
  onReset: () => void;
  speed: number;
  onSpeedChange: (speed: number) => void;
}

const STATUS_BADGE: Record<HealthStatus, string> = {
  Healthy: "bg-emerald-500/15 text-emerald-300 border-emerald-500/30",
  Warning: "bg-amber-500/15 text-amber-300 border-amber-500/30",
  Critical: "bg-rose-500/15 text-rose-300 border-rose-500/30",
};

export function EngineControls({
  engines,
  selectedEngineId,
  onSelectEngine,
  currentCycle,
  maxCycle,
  onScrubCycle,
  isPlaying,
  onTogglePlay,
  onReset,
  speed,
  onSpeedChange,
}: Props) {
  const criticalEngines = engines.filter((e) => e.health_status === "Critical").slice(0, 5);
  const selectedEngine = engines.find((e) => e.engine_id === selectedEngineId);

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/80 p-4 shadow-lg">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
        {/* Engine Selector */}
        <div className="flex flex-wrap items-center gap-3">
          <div>
            <label className="block text-[11px] font-semibold uppercase tracking-wider text-slate-400">
              Select Fleet Asset
            </label>
            <select
              value={selectedEngineId}
              onChange={(e) => onSelectEngine(Number(e.target.value))}
              className="mt-1 rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 font-mono text-sm text-slate-100 focus:border-sky-500 focus:outline-none"
            >
              {engines.map((eng) => (
                <option key={eng.engine_id} value={eng.engine_id}>
                  Engine #{String(eng.engine_id).padStart(3, "0")} — RUL {eng.predicted_rul.toFixed(0)} cyc ({eng.health_status})
                </option>
              ))}
            </select>
          </div>

          {selectedEngine && (
            <span
              className={`mt-4 inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-semibold ${
                STATUS_BADGE[selectedEngine.health_status]
              }`}
            >
              <span className="h-1.5 w-1.5 rounded-full bg-current" />
              Final State: {selectedEngine.health_status}
            </span>
          )}

          {criticalEngines.length > 0 && (
            <div className="hidden items-center gap-1.5 pt-4 xl:flex">
              <span className="text-[11px] uppercase tracking-wider text-slate-500">
                Critical Quick-Jump:
              </span>
              {criticalEngines.map((ce) => (
                <button
                  key={ce.engine_id}
                  onClick={() => onSelectEngine(ce.engine_id)}
                  className={`rounded-md border px-2 py-1 font-mono text-xs transition ${
                    ce.engine_id === selectedEngineId
                      ? "border-rose-400 bg-rose-500/20 text-rose-200"
                      : "border-slate-700 bg-slate-950 text-slate-300 hover:border-rose-500/50 hover:text-rose-300"
                  }`}
                >
                  #{ce.engine_id}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Live Telemetry Playback Buttons */}
        <div className="flex flex-wrap items-center gap-2.5">
          <button
            onClick={onTogglePlay}
            className={`inline-flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-semibold transition ${
              isPlaying
                ? "bg-amber-500 text-slate-950 hover:bg-amber-400"
                : "bg-sky-500 text-slate-950 hover:bg-sky-400"
            }`}
          >
            {isPlaying ? (
              <>
                <Pause className="h-4 w-4" />
                Pause Stream
              </>
            ) : (
              <>
                <Play className="h-4 w-4" />
                Play Live Telemetry
              </>
            )}
          </button>

          <button
            onClick={onReset}
            title="Reset to cycle 1"
            className="inline-flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-xs font-medium text-slate-300 transition hover:border-slate-500 hover:text-white"
          >
            <RotateCcw className="h-3.5 w-3.5" />
            Reset
          </button>

          <div className="flex items-center gap-1 rounded-lg border border-slate-800 bg-slate-950 p-1">
            {[1, 2, 4].map((s) => (
              <button
                key={s}
                onClick={() => onSpeedChange(s)}
                className={`rounded px-2 py-1 font-mono text-xs transition ${
                  speed === s
                    ? "bg-slate-800 text-sky-400"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                {s}x
              </button>
            ))}
          </div>

          {isPlaying && (
            <span className="inline-flex items-center gap-1.5 rounded-md bg-emerald-500/10 px-2.5 py-1 font-mono text-xs text-emerald-400">
              <Radio className="h-3.5 w-3.5 animate-pulse" />
              SSE LIVE
            </span>
          )}
        </div>
      </div>

      {/* Cycle Scrubber Slider */}
      <div className="mt-4 border-t border-slate-800/80 pt-3">
        <div className="mb-1.5 flex items-center justify-between text-xs">
          <span className="font-medium text-slate-400">
            Flight Cycle Timeline Scrubber
          </span>
          <span className="font-mono text-sky-300">
            Cycle {currentCycle} / {maxCycle || "—"}
          </span>
        </div>
        <input
          type="range"
          min={1}
          max={Math.max(1, maxCycle)}
          value={currentCycle}
          onChange={(e) => onScrubCycle(Number(e.target.value))}
          className="h-2 w-full cursor-pointer appearance-none rounded-lg bg-slate-800 accent-sky-400"
        />
      </div>
    </div>
  );
}
