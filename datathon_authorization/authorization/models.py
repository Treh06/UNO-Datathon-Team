"""Request/response definitions for synthetic purchases only."""
from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum
from ipaddress import ip_network
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class CardType(str, Enum):
    credit = "credit"
    debit = "debit"
    prepaid = "prepaid"
    virtual = "virtual"


class Transaction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    txn_id: str = Field(pattern=r"^T[0-9]{10}$")
    event_ts: datetime
    card_token: str = Field(pattern=r"^tok_[A-Za-z0-9_]+$")
    bin: str = Field(pattern=r"^[0-9]{6}$")  # synthetic BIN-like value only
    card_type: CardType
    card_issue_date: date
    merchant_id: str = Field(min_length=1, max_length=64)
    mcc: str = Field(pattern=r"^[0-9]{4}$")
    event_id: str | None = None
    product_category: Literal["tickets", "sneakers", "cards", "collectibles", "other"]
    quantity: int = Field(ge=1, le=100)
    amount: Decimal = Field(gt=Decimal("0"), max_digits=12, decimal_places=2)
    currency: Literal["USD"] = "USD"
    ship_addr_hash: str = Field(min_length=1)
    bill_addr_hash: str = Field(min_length=1)
    email_hash: str = Field(min_length=1)
    device_id: str | None = None
    ip_prefix: str | None = None
    channel: Literal["web", "app", "api"]

    @field_validator("event_ts")
    @classmethod
    def utc_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != timezone.utc.utcoffset(value):
            raise ValueError("event_ts must be a timezone-aware UTC timestamp")
        return value.astimezone(timezone.utc)

    @field_validator("ip_prefix")
    @classmethod
    def valid_prefix(cls, value: str | None) -> str | None:
        if value is not None:
            ip_network(value, strict=False)
        return value

    @model_validator(mode="after")
    def issue_date_is_valid(self):
        if self.card_issue_date > self.event_ts.date():
            raise ValueError("card_issue_date cannot be later than event_ts")
        return self


class AuthorizationResponse(BaseModel):
    txn_id: str
    authorization_status: Literal["APPROVE", "REVIEW", "DECLINE"]
    risk_score: int = Field(ge=0, le=100)
    reason_codes: list[str]
    processed_ts: datetime
