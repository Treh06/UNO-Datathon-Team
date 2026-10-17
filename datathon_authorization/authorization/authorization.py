"""Illustrative risk rules. No bank, card network, or payment processor is contacted."""
from datetime import timedelta, datetime, timezone

from authorization.models import AuthorizationResponse, Transaction

# Example per-buyer limits; change these to match an actual drop's rules.
LIMITS = {"tickets": 4, "sneakers": 1, "cards": 2, "collectibles": 2, "other": 2}
WINDOW = timedelta(seconds=60)


def authorize(txn: Transaction, history: list[Transaction]) -> AuthorizationResponse:
    """Use only earlier requests for this drop; score rules are illustrative."""
    same_drop = [t for t in history if t.merchant_id == txn.merchant_id
                 and t.event_id == txn.event_id and t.event_ts <= txn.event_ts]
    recent = [t for t in same_drop if txn.event_ts - t.event_ts <= WINDOW]
    reasons: list[str] = []
    score = 0

    # Count previous purchases using same account or payment token for this drop.
    buyer_prior = [t for t in same_drop if t.email_hash == txn.email_hash
                   or t.card_token == txn.card_token]
    prior_quantity = sum(t.quantity for t in buyer_prior)
    if prior_quantity + txn.quantity > LIMITS[txn.product_category]:
        reasons.append("QUANTITY_LIMIT_EXCEEDED")
        score += 60

    # Velocity: repeated orders by one buyer within a minute.
    buyer_recent = [t for t in recent if t.email_hash == txn.email_hash
                    or t.card_token == txn.card_token]
    if len(buyer_recent) >= 3:
        reasons.append("HIGH_PURCHASE_VELOCITY")
        score += 25

    # Multiple different card tokens from one device during a drop.
    if txn.device_id:
        cards_on_device = {t.card_token for t in recent if t.device_id == txn.device_id}
        cards_on_device.add(txn.card_token)
        if len(cards_on_device) >= 3:
            reasons.append("MULTIPLE_CARDS_SAME_DEVICE")
            score += 30

    # Multiple email identities at the same shipping address.
    emails_at_address = {t.email_hash for t in same_drop
                         if t.ship_addr_hash == txn.ship_addr_hash}
    emails_at_address.add(txn.email_hash)
    if len(emails_at_address) >= 3:
        reasons.append("SHARED_SHIPPING_ADDRESS")
        score += 20

    card_age_days = (txn.event_ts.date() - txn.card_issue_date).days
    if card_age_days <= 1:
        reasons.append("NEWLY_ISSUED_CARD")
        score += 10

    score = min(100, score)
    # Only explicit quantity violation auto-declines in this demonstration.
    # Other indicators alone lead to REVIEW, never imply proven fraud.
    status = ("DECLINE" if "QUANTITY_LIMIT_EXCEEDED" in reasons else
              "REVIEW" if score >= 30 else "APPROVE")
    return AuthorizationResponse(
        txn_id=txn.txn_id, authorization_status=status, risk_score=score,
        reason_codes=reasons, processed_ts=datetime.now(timezone.utc)
    )
