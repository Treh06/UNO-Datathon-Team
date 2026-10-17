"""Replay the provided synthetic examples into the local API."""
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

samples = json.loads((Path(__file__).parent / "authorization" / "sample_transactions.json").read_text())
for txn in samples:
    payload = json.dumps(txn).encode()
    request = Request("http://127.0.0.1:8000/authorize", data=payload,
                      headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(request) as response:
            print(json.loads(response.read()))
    except HTTPError as exc:
        print(exc.code, exc.read().decode())
