"""Map a trained ticket detector's advisory action to a simulated status."""

from datetime import datetime, timezone

from .models import AuthorizationResponse, Transaction
from .validator import validate_transaction


_STATUS = {
    "allow": "APPROVE",
    "step_up_verification": "REVIEW",
    "quantity_cap_or_throttle": "REVIEW",
    "cancel_or_refund": "DECLINE",
}


def authorize(txn: Transaction, session, *, order: dict | None = None) -> AuthorizationResponse:
    """Score the observed request using the saved detector, without rule fallback.

    All outcomes remain simulated. Requests are observed attempts; neither
    this function nor its status executes an actual payment or merchant action.
    """
    prepared = validate_transaction(txn) if order is None else order
    result = session.score_order(prepared)
    action = result["recommended_action"]
    return AuthorizationResponse(
        txn_id=txn.txn_id,
        authorization_status=_STATUS[action],
        risk_score=int(round(float(result["risk_score"]))),
        reason_codes=result["reason_codes"],
        processed_ts=datetime.now(timezone.utc),
        scalper_probability=result["scalper_probability"],
        confidence_tier=result["confidence_tier"],
        recommended_action=action,
        flagged_for_review=result["flagged_for_review"],
        decision_threshold=result["decision_threshold"],
        model_result=result,
    )
