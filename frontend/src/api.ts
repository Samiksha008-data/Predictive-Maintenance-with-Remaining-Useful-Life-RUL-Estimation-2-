import {
  EngineDetailResponse,
  EngineSummary,
  ExplainCycleResponse,
  OverviewMetrics,
  TelemetryPoint,
} from "./types";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

async function fetchJson<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API ${path} failed (${res.status}): ${text}`);
  }
  return res.json() as Promise<T>;
}

export function getOverview(): Promise<OverviewMetrics> {
  return fetchJson<OverviewMetrics>("/api/overview");
}

export function getEngines(): Promise<EngineSummary[]> {
  return fetchJson<EngineSummary[]>("/api/engines");
}

export function getEngineDetail(engineId: number): Promise<EngineDetailResponse> {
  return fetchJson<EngineDetailResponse>(`/api/engines/${engineId}`);
}

export function getCycleExplanation(
  engineId: number,
  cycle: number
): Promise<ExplainCycleResponse> {
  return fetchJson<ExplainCycleResponse>(`/api/engines/${engineId}/explain/${cycle}`);
}

export function subscribeTelemetryStream(
  engineId: number,
  startCycle: number,
  delaySeconds: number,
  onPoint: (point: TelemetryPoint) => void,
  onComplete: () => void,
  onError: (err: string) => void
): () => void {
  const url = `${API_BASE}/api/simulate/${engineId}?start_cycle=${startCycle}&delay=${delaySeconds}`;
  const source = new EventSource(url);

  source.onmessage = (event) => {
    try {
      const point = JSON.parse(event.data) as TelemetryPoint;
      onPoint(point);
      if (point.done) {
        source.close();
        onComplete();
      }
    } catch (e) {
      onError(e instanceof Error ? e.message : "Failed to parse SSE telemetry");
    }
  };

  source.onerror = () => {
    source.close();
    onError("Telemetry stream connection interrupted");
  };

  return () => {
    source.close();
  };
}
