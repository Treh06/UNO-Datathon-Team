"""Business-independent validation beyond the Pydantic field types."""
from authorization.models import Transaction


def validate_transaction(txn: Transaction) -> None:
    """Reject invalid data; suspicious behavior is handled by authorization.py."""
    if not txn.event_id:
        raise ValueError("event_id is required for limited-release purchases")
    if not txn.event_id.strip():
        raise ValueError("event_id cannot be blank")
    for name in ("ship_addr_hash", "bill_addr_hash", "email_hash"):
        if not getattr(txn, name).strip():
            raise ValueError(f"{name} cannot be blank")
