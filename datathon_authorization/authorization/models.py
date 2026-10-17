"""Ticket authorization requests and simulated trained-model responses."""

from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from enum import Enum
from ipaddress import ip_network
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator


class CardType(str, Enum):
    credit = "credit"
    debit = "debit"
    prepaid = "prepaid"
    virtual = "virtual"


class Transaction(BaseModel):
    """Only information observable on a ticket authorization request.

    Ground-truth/persona fields are forbidden by the extra-field policy. The
    generator omits product_category; its supported ticket MCC supplies scope.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, allow_inf_nan=False)

    txn_id: str = Field(strict=True, min_length=2, max_length=128, pattern=r"^T[0-9]+$")
    event_ts: datetime
    card_token: str = Field(strict=True, min_length=1, max_length=256)
    bin: str = Field(strict=True, pattern=r"^[0-9]{6}$")
    card_type: CardType | None = None
    card_issue_date: date | None = None
    merchant_id: str = Field(strict=True, min_length=1, max_length=128)
    mcc: Literal["7922"]
    event_id: str | None = Field(default=None, strict=True, max_length=256)
    product_category: Literal["tickets"] = "tickets"
    quantity: StrictInt = Field(ge=1)
    amount: Decimal = Field(gt=Decimal("0"), max_digits=12, decimal_places=2)
    currency: Literal["USD"]
    ship_addr_hash: str | None = Field(default=None, strict=True, max_length=256)
    bill_addr_hash: str | None = Field(default=None, strict=True, max_length=256)
    email_hash: str | None = Field(default=None, strict=True, max_length=256)
    device_id: str | None = Field(default=None, strict=True, max_length=256)
    ip_prefix: str | None = Field(default=None, strict=True, max_length=128)
    channel: Literal["web", "app", "api"] | None = None

    @field_validator("mcc", "currency", "product_category", "card_type", "channel", mode="before")
    @classmethod
    def strip_codes(cls, value):
        return value.strip() if isinstance(value, str) else value

    @field_validator("event_ts", mode="before")
    @classmethod
    def explicit_timestamp(cls, value):
        if not isinstance(value, (str, datetime)):
            raise ValueError("event_ts must be an explicit timezone-aware ISO timestamp")
        return value.strip() if isinstance(value, str) else value

    @field_validator("event_ts")
    @classmethod
    def utc_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("event_ts must include a timezone")
        return value.astimezone(timezone.utc)

    @field_validator("amount", mode="before")
    @classmethod
    def finite_amount(cls, value):
        if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
            raise ValueError("amount must be a finite positive decimal amount")
        try:
            amount = Decimal(str(value))
        except (InvalidOperation, ValueError) as exc:
            raise ValueError("amount must be a finite positive decimal amount") from exc
        if not amount.is_finite() or amount <= 0:
            raise ValueError("amount must be a finite positive decimal amount")
        return amount

    @field_validator("card_issue_date", mode="before")
    @classmethod
    def explicit_issue_date(cls, value):
        if value in (None, ""):
            return None
        if isinstance(value, datetime) or not isinstance(value, (str, date)):
            raise ValueError("card_issue_date must be a YYYY-MM-DD date")
        if isinstance(value, str):
            value = value.strip()
            try:
                parsed = date.fromisoformat(value)
            except ValueError as exc:
                raise ValueError("card_issue_date must be a YYYY-MM-DD date") from exc
            if parsed.isoformat() != value:
                raise ValueError("card_issue_date must be a YYYY-MM-DD date")
        return value

    @field_validator("event_id", "ship_addr_hash", "bill_addr_hash", "email_hash", "device_id", "ip_prefix")
    @classmethod
    def optional_tokens(cls, value: str | None) -> str | None:
        return value or None

    @field_validator("ip_prefix")
    @classmethod
    def valid_prefix(cls, value: str | None) -> str | None:
        if value is not None:
            ip_network(value, strict=False)
        return value

    @model_validator(mode="after")
    def issue_date_is_valid(self):
        if self.card_issue_date is not None and self.card_issue_date > self.event_ts.date():
            raise ValueError("card_issue_date cannot be later than event_ts in UTC")
        return self


class AuthorizationResponse(BaseModel):
    """Advisory simulation output; no payment or cancellation is executed."""

    model_config = ConfigDict(allow_inf_nan=False)

    txn_id: str
    authorization_status: Literal["APPROVE", "REVIEW", "DECLINE"]
    risk_score: int = Field(ge=0, le=100)
    reason_codes: list[str]
    processed_ts: datetime
    scalper_probability: float = Field(ge=0, le=1)
    confidence_tier: Literal["limited", "low", "moderate", "high"]
    recommended_action: Literal["allow", "step_up_verification", "quantity_cap_or_throttle", "cancel_or_refund"]
    flagged_for_review: bool
    decision_threshold: float = Field(ge=0, le=1)
    model_result: dict[str, Any]
