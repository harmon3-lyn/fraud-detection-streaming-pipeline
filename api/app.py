"""
api/app.py - Scoring endpoint (FastAPI)

Run with:
    uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET /health (liveness check)
    POST /score (score single transaction)
    POST /score/batch (score list of transactions)
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from score_transaction import score_transaction

app = FastAPI(title="Fraud Detection API", version="1.0")


class TransactionIn(BaseModel):
    trans_num: str = ""
    cc_num: int = 0
    merchant: str = ""
    unix_time: int = 0
    amt: float = 0.0
    is_fraud: int = -1 # ground truth; -1 = unknown (production)
    model_config = {"extra": "allow"}


class DecisionOut(BaseModel):
    trans_num: str
    fraud_score: float
    decision: str
    scored_at: str
    amt: float
    is_fraud_actual: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/score", response_model=DecisionOut)
def score(txn: TransactionIn):
    try:
        result = score_transaction(txn.model_dump(), write_to_sql=True)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/score/batch", response_model=list[DecisionOut])
def score_batch(txns: list[TransactionIn]):
    results = []
    for txn in txns:
        try:
            results.append(score_transaction(txn.model_dump(), write_to_sql=True))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))
    return results
