"""Verification script for all FastAPI endpoints and SSE telemetry streaming."""
import numpy as np

from backend.main import (
    PredictRequest,
    explain_engine_cycle,
    get_engine_detail,
    get_overview,
    health,
    list_engines,
    predict,
    simulate,
    state,
)


def _step(coro):
    try:
        coro.send(None)
    except StopIteration as exc:
        return exc.value
    raise RuntimeError("Unexpected suspension in synchronous verification")


def main() -> None:
    print("1. Loading backend artifacts & engine cache...", flush=True)
    state.load()
    print("2. HEALTH:", _step(health()), flush=True)
    print("3. OVERVIEW:", _step(get_overview()), flush=True)

    engines = _step(list_engines())
    e24 = next(e for e in engines if e["engine_id"] == 24)
    print(f"4. ENGINES COUNT: {len(engines)} | ENGINE #24: {e24}", flush=True)

    detail = _step(get_engine_detail(24))
    print(
        f"5. DETAIL #24: cycles={detail['total_cycles']} latest_pred={detail['latest']['predicted_rul']}",
        flush=True,
    )

    exp = _step(explain_engine_cycle(24, 50))
    print("6. EXPLAIN #24 CYCLE 50:", exp, flush=True)

    X_test = np.load("ml_pipeline/artifacts/X_test_seq.npy", mmap_mode="r")
    y_test = np.load("ml_pipeline/artifacts/y_test_seq.npy")
    pred = _step(predict(PredictRequest(window=X_test[0].tolist(), true_rul=float(y_test[0]))))
    print("7. PREDICT:", pred, flush=True)

    resp = _step(simulate(24, delay=0.0, start_cycle=1))
    events = []
    agen = resp.body_iterator
    while True:
        try:
            events.append(_step(agen.__anext__()))
        except StopAsyncIteration:
            break
    print(f"8. SSE EVENTS FOR #24: {len(events)} | LAST: {events[-1][:180]}", flush=True)
    print("ALL BACKEND CHECKS PASSED!", flush=True)


if __name__ == "__main__":
    main()
