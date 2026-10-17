"""Adapt validated request data before any chronological scoring-state update."""

from scalper_detector.authorizations import authorization_to_order

from .models import Transaction


def validate_transaction(txn: Transaction) -> dict:
    """Return a canonical label-free order, retaining missing-data flags.

    Missing performance or linkage tokens remain unknown. Neither a category
    purchase limit nor a release time is invented for the authorization API.
    """
    return authorization_to_order(txn.model_dump(mode="json"))
