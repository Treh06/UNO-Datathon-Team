"""Single-worker, local authorization simulation backed by a saved detector.

Run from the repository root with:
    uvicorn datathon_authorization.authorization.main:app
"""

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import json
import logging
import math
import os
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import torch

from .authorization import authorize
from .models import AuthorizationResponse, Transaction
from .validator import validate_transaction


_LOGGER = logging.getLogger(__name__)


def _validation_float(value: float) -> float | str:
    """Keep rejected nonfinite JSON numbers serializable in error details."""
    if math.isfinite(value):
        return value
    if math.isnan(value):
        return "NaN"
    return "Infinity" if value > 0 else "-Infinity"


@dataclass
class _AuthorizationState:
    detector: Any
    session: Any
    model_directory: str
    ready: bool = True
    failure: str | None = None
    seen_ids: set[str] = field(default_factory=set)
    decisions: dict[str, int] = field(default_factory=lambda: {"APPROVE": 0, "REVIEW": 0, "DECLINE": 0})
    last_event_ts: Any = None
    lock: Any = field(default_factory=Lock)


def create_app(*, model_dir: str | Path | None = None, detector=None) -> FastAPI:
    """Load one saved detector during lifespan and keep state isolated per app.

    ``detector`` injection supports integration tests. Production startup uses
    ``model_dir`` or SCALPER_MODEL_DIR, defaulting to artifacts/ticket_model.
    Run a single worker: state is chronological, in memory, and not shared.
    """
    directory = Path(model_dir if model_dir is not None else os.environ.get("SCALPER_MODEL_DIR", "artifacts/ticket_model"))

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        try:
            if detector is None:
                from scalper_detector.pipeline import ScalperDetector
                loaded = ScalperDetector.load(directory)
            else:
                loaded = detector
            threads = loaded.metadata.get("training_config", {}).get("threads", 1)
            if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
                raise ValueError("Saved training_config.threads must be a positive integer.")
            torch.set_num_threads(threads)
            session = loaded.new_session()
        except Exception as exc:
            raise RuntimeError(
                f"Cannot start authorization API: a saved ticket detector is required at {directory}. "
                "Train the detector first or set SCALPER_MODEL_DIR to the trained model directory. "
                f"Load failure: {exc}"
            ) from exc
        state = _AuthorizationState(loaded, session, str(directory.resolve()))
        application.state.authorization_state = state
        try:
            yield
        finally:
            with state.lock:
                state.ready = False
                state.session = None

    application = FastAPI(title="Trained Ticket Authorization Simulation", version="2.0.0", lifespan=lifespan)
    application.state.authorization_state = None

    @application.exception_handler(RequestValidationError)
    async def invalid_request(_request, exc: RequestValidationError):
        # Python's JSON decoder can accept NaN/Infinity. The request schema
        # rejects these, but echoing them unchanged in validation errors would
        # make JSONResponse raise and turn a client error into HTTP 500.
        detail = jsonable_encoder(exc.errors(), custom_encoder={float: _validation_float})
        return JSONResponse(status_code=422, content={"detail": detail})

    def state_or_unavailable() -> _AuthorizationState:
        state = application.state.authorization_state
        if state is None:
            raise HTTPException(status_code=503, detail="The ticket detector is not ready.")
        return state

    @application.get("/health")
    def health():
        state = application.state.authorization_state
        if state is None:
            return JSONResponse(status_code=503, content={"status": "unavailable", "ready": False, "mode": "simulation"})
        with state.lock:
            metadata = state.detector.metadata
            result = {
                "status": "ok" if state.ready else "unavailable",
                "ready": state.ready,
                "mode": "simulation",
                "single_worker_required": True,
                "state": "in_memory; restart resets chronological context",
                "merchant_policy": "default detector policy; exact cutoffs are in model_result",
                "model": {
                    "directory": state.model_directory,
                    "task": metadata.get("task"),
                    "format_version": metadata.get("format_version"),
                    "data_source": metadata.get("data_source", "unknown"),
                    "input_size": metadata.get("model_config", {}).get("input_size"),
                    "sequence_length": metadata.get("training_config", {}).get("sequence_length"),
                    "cpu_threads": metadata.get("training_config", {}).get("threads", 1),
                },
            }
            if state.failure:
                result["detail"] = state.failure
            return JSONResponse(status_code=200 if state.ready else 503, content=result)

    @application.post("/authorize", response_model=AuthorizationResponse)
    def authorize_request(txn: Transaction):
        # Pydantic and adapter validation finish before touching shared state.
        try:
            order = validate_transaction(txn)
        except (ValueError, TypeError, OverflowError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        state = state_or_unavailable()
        with state.lock:
            if not state.ready:
                raise HTTPException(status_code=503, detail=state.failure or "The ticket detector is not ready.")
            if txn.txn_id in state.seen_ids:
                raise HTTPException(status_code=409, detail="txn_id already processed")
            if state.last_event_ts is not None and txn.event_ts < state.last_event_ts:
                raise HTTPException(status_code=409, detail="event_ts is out of order; requests must be globally chronological")
            try:
                response = authorize(txn, state.session, order=order)
                # Ensure serialization also succeeds before committing API IDs
                # or counters. The scoring session may already have observed
                # this attempt, so an unexpected failure must stop the service.
                json.dumps(response.model_dump(mode="json"), allow_nan=False)
            except Exception as exc:
                state.ready = False
                state.failure = "Ticket inference failed; restart the service before submitting more requests."
                state.session = None
                _LOGGER.exception("Ticket authorization inference failed; chronological session discarded")
                raise HTTPException(status_code=503, detail=state.failure) from exc
            state.seen_ids.add(txn.txn_id)
            state.last_event_ts = txn.event_ts
            state.decisions[response.authorization_status] += 1
            return response

    @application.get("/stats")
    def stats():
        state = state_or_unavailable()
        with state.lock:
            return {"requests_processed": len(state.seen_ids), "decisions": dict(state.decisions)}

    return application


app = create_app()
