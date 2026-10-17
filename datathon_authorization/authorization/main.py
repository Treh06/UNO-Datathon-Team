"""Run with: uvicorn authorization.main:app --reload"""
from threading import Lock

from fastapi import FastAPI, HTTPException

from authorization.authorization import authorize
from authorization.models import AuthorizationResponse, Transaction
from authorization.validator import validate_transaction

app = FastAPI(title="Synthetic Purchase Authorization", version="1.0.0")
_history: list[Transaction] = []
_results: dict[str, AuthorizationResponse] = {}
_lock = Lock()


@app.get("/health")
def health():
    return {"status": "ok", "mode": "simulation"}


@app.post("/authorize", response_model=AuthorizationResponse)
def authorize_request(txn: Transaction):
    try:
        validate_transaction(txn)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    with _lock:
        if txn.txn_id in _results:
            # Prevent repeated identical order IDs from changing the history.
            raise HTTPException(status_code=409, detail="txn_id already processed")
        decision = authorize(txn, _history)
        _history.append(txn)  # Demo treats every submitted request as an attempt.
        _results[txn.txn_id] = decision
    return decision


@app.get("/stats")
def stats():
    with _lock:
        counts = {status: sum(result.authorization_status == status
                              for result in _results.values())
                  for status in ("APPROVE", "REVIEW", "DECLINE")}
        return {"requests_processed": len(_history), "decisions": counts}
