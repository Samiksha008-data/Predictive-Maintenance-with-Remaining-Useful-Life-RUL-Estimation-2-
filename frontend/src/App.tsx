import { useEffect, useRef, useState } from "react";
import { Plane, RefreshCw } from "lucide-react";
import {
  getCycleExplanation,
  getEngineDetail,
  getEngines,
  getOverview,
  subscribeTelemetryStream,
} from "./api";
import { DegradationChart } from "./components/DegradationChart";
import { EngineControls } from "./components/EngineControls";
import { ExplainabilityPanel } from "./components/ExplainabilityPanel";
import { FleetOverviewCards } from "./components/FleetOverviewCards";
import { SensorTelemetryChart } from "./components/SensorTelemetryChart";
import {
  EngineSummary,
  ExplainCycleResponse,
  OverviewMetrics,
  TelemetryPoint,
} from "./types";

export default function App() {
  const [overview, setOverview] = useState<OverviewMetrics | null>(null);
  const [engines, setEngines] = useState<EngineSummary[]>([]);
  const [selectedEngineId, setSelectedEngineId] = useState<number>(24);
  const [fullTrajectory, setFullTrajectory] = useState<TelemetryPoint[]>([]);
  const [visibleTrajectory, setVisibleTrajectory] = useState<TelemetryPoint[]>([]);
  const [currentCycle, setCurrentCycle] = useState<number>(1);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [speed, setSpeed] = useState<number>(2);
  const [explanation, setExplanation] = useState<ExplainCycleResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  const stopStreamRef = useRef<(() => void) | null>(null);

  const stopStream = () => {
    if (stopStreamRef.current) {
      stopStreamRef.current();
      stopStreamRef.current = null;
    }
    setIsPlaying(false);
  };

  // Load initial fleet overview and engine list
  useEffect(() => {
    let active = true;
    async function init() {
      try {
        setLoading(true);
        const [ov, engList] = await Promise.all([getOverview(), getEngines()]);
        if (!active) return;
        setOverview(ov);
        setEngines(engList);
        // Prefer Engine #24 or first Critical engine if present
        const preferred =
          engList.find((e) => e.engine_id === 24) ??
          engList.find((e) => e.health_status === "Critical") ??
          engList[0];
        if (preferred) {
          setSelectedEngineId(preferred.engine_id);
        }
        setError(null);
      } catch (e) {
        if (!active) return;
        setError(e instanceof Error ? e.message : "Failed to connect to FastAPI backend");
      } finally {
        if (active) setLoading(false);
      }
    }
    init();
    return () => {
      active = false;
      stopStream();
    };
  }, []);

  // Load selected engine full trajectory for scrubbing
  useEffect(() => {
    let active = true;
    stopStream();

    async function loadEngine() {
      if (!selectedEngineId) return;
      try {
        const detail = await getEngineDetail(selectedEngineId);
        if (!active) return;
        setFullTrajectory(detail.trajectory);
        setVisibleTrajectory(detail.trajectory);
        setCurrentCycle(detail.total_cycles);
        setExplanation({
          engine_id: selectedEngineId,
          cycle: detail.latest.cycle,
          predicted_rul: detail.latest.predicted_rul,
          health_status: detail.latest.health_status,
          top_contributing_sensors: detail.latest.top_contributing_sensors ?? [],
          maintenance_recommendation: detail.latest.maintenance_recommendation ?? "",
        });
        setError(null);
      } catch (e) {
        if (!active) return;
        setError(e instanceof Error ? e.message : "Failed to fetch engine trajectory");
      }
    }
    loadEngine();
    return () => {
      active = false;
    };
  }, [selectedEngineId]);

  // Fetch live SHAP explanation when scrubbing manually (when paused)
  const handleScrubCycle = (cycle: number) => {
    stopStream();
    setCurrentCycle(cycle);
    setVisibleTrajectory(fullTrajectory.slice(0, cycle));
    getCycleExplanation(selectedEngineId, cycle)
      .then((exp) => setExplanation(exp))
      .catch(() => {});
  };

  const handleTogglePlay = () => {
    if (isPlaying) {
      stopStream();
      return;
    }

    const startCycle = currentCycle >= fullTrajectory.length ? 1 : currentCycle;
    if (startCycle === 1) {
      setVisibleTrajectory([]);
      setCurrentCycle(1);
    }

    const delaySeconds = speed === 4 ? 0.025 : speed === 2 ? 0.05 : 0.1;
    setIsPlaying(true);

    stopStreamRef.current = subscribeTelemetryStream(
      selectedEngineId,
      startCycle,
      delaySeconds,
      (point) => {
        setCurrentCycle(point.cycle);
        setVisibleTrajectory((prev) => {
          const base = prev.filter((p) => p.cycle < point.cycle);
          return [...base, point];
        });
        if (point.top_contributing_sensors) {
          setExplanation({
            engine_id: point.engine_id,
            cycle: point.cycle,
            predicted_rul: point.predicted_rul,
            health_status: point.health_status,
            top_contributing_sensors: point.top_contributing_sensors,
            maintenance_recommendation: point.maintenance_recommendation ?? "",
          });
        }
      },
      () => {
        setIsPlaying(false);
        stopStreamRef.current = null;
      },
      (errMsg) => {
        setIsPlaying(false);
        stopStreamRef.current = null;
        setError(errMsg);
      }
    );
  };

  const handleReset = () => {
    stopStream();
    if (fullTrajectory.length > 0) {
      setCurrentCycle(1);
      setVisibleTrajectory(fullTrajectory.slice(0, 1));
      getCycleExplanation(selectedEngineId, 1)
        .then((exp) => setExplanation(exp))
        .catch(() => {});
    }
  };

  const currentPoint =
    visibleTrajectory[visibleTrajectory.length - 1] ??
    fullTrajectory[currentCycle - 1] ??
    null;

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      {/* Top Aerospace Header */}
      <header className="border-b border-slate-800/80 bg-slate-950/90 backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3.5 sm:px-6">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg border border-sky-500/30 bg-sky-500/10 text-sky-400">
              <Plane className="h-5 w-5" />
            </div>
            <div>
              <h1 className="text-base font-bold tracking-tight text-white sm:text-lg">
                NASA C-MAPSS Fleet Telemetry &amp; RUL Prognostics
              </h1>
              <p className="text-xs text-slate-400">
                Turbofan Predictive Maintenance · FD001 · Asymmetric Risk &amp; SHAP Explainability
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <span className="hidden rounded-full border border-slate-800 bg-slate-900 px-3 py-1 font-mono text-xs text-slate-300 sm:inline-block">
              {overview ? `${overview.best_model.toUpperCase()} ACTIVE` : "CONNECTING..."}
            </span>
            {loading && <RefreshCw className="h-4 w-4 animate-spin text-sky-400" />}
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl space-y-5 px-4 py-5 sm:px-6">
        {error && (
          <div className="rounded-xl border border-rose-500/40 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
            <strong>Backend Connection Notice:</strong> {error}. Make sure FastAPI is running on{" "}
            <code className="rounded bg-slate-900 px-1.5 py-0.5 font-mono text-xs">
              http://127.0.0.1:8000
            </code>
            .
          </div>
        )}

        <FleetOverviewCards overview={overview} />

        <EngineControls
          engines={engines}
          selectedEngineId={selectedEngineId}
          onSelectEngine={(id) => setSelectedEngineId(id)}
          currentCycle={currentCycle}
          maxCycle={fullTrajectory.length}
          onScrubCycle={handleScrubCycle}
          isPlaying={isPlaying}
          onTogglePlay={handleTogglePlay}
          onReset={handleReset}
          speed={speed}
          onSpeedChange={setSpeed}
        />

        <div className="grid grid-cols-1 gap-5 xl:grid-cols-2">
          <DegradationChart data={visibleTrajectory} />
          <SensorTelemetryChart data={visibleTrajectory} />
        </div>

        <ExplainabilityPanel currentPoint={currentPoint} explanation={explanation} />
      </main>
    </div>
  );
}
