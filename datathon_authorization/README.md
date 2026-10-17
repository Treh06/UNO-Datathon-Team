# Limited-Release Transaction Authorization Simulator

**Synthetic data only.** This API does not charge cards or contact banks. It gives illustrative APPROVE, REVIEW, and DECLINE decisions for limited-release tickets, sneakers, cards, collectibles, and other items. `bin` contains *synthetic six-digit placeholders*, not real issuer data.

## Setup

From the project root, with Python 3.10+:

```bash
python -m venv .venv
```

Activate your virtual environment:
- Windows PowerShell: `.venv\Scripts\Activate.ps1`
- macOS/Linux: `source .venv/bin/activate`

```bash
pip install -r requirements.txt
uvicorn authorization.main:app --reload
```

Open **http://127.0.0.1:8000/docs** to test transactions in Swagger UI.

## Replay samples in order

In a second terminal, from the project root:

```bash
python replay.py
```

The first concert ticket purchase should be APPROVE. The second uses the same card and exceeds the four-ticket cumulative limit, so it should be DECLINE. Replaying identical IDs triggers HTTP 409; restart the server to reset the in-memory state.

## Endpoints

- `POST /authorize` — JSON purchase request => authorization response
- `GET /health` — availability
- `GET /stats` — demo decision counts

## Important scope notes

- This is a classroom demonstration, not a real card network/payment authorization integration.
- State is in memory and resets on restart; run a **single server worker**.
- Incoming timestamps are simulated UTC event times; submit records in chronological order. Out-of-order events do not retroactively change previous decisions.
- The simulator records all *attempted* requests (including declines) as prior activity; production order limits should normally use confirmed/completed purchases, and attempts should be tracked separately.
- Merchant/drop-specific limits must be agreed with the team. Current category limits are sample values.
- A shared address, new card, or high rate is not proof of scalping. Review scores are illustrative, not trained or calibrated.
- The API has no authentication and should only run locally in a trusted demo environment.
