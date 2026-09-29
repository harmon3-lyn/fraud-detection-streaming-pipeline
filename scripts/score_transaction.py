"""
Accepts pre-computed feature dict, scores it with trained LightGBM model,
writes the decision to fraud_decisions, and returns a result dict.
"""

import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
sys.path.insert(0, str(BASE_DIR / "scripts"))


# Decision threshold
REVIEW_THRESHOLD = 0.30   # score >= REVIEW_THRESHOLD and < DECLINE_THRESHOLD => REVIEW

# Categorical cols
CAT_COLS = [
    "IP_Country", 
    "MerchantCategory", 
    "MerchantCountry",
    "CarrierType", 
    "IdentityVerificationStatus",
]

_model = None
_encoders = None
_feature_cols = None
_threshold = None


def _load_artifacts():
    global _model, _encoders, _feature_cols, _threshold
    if _model is not None:
        return
    with open(MODEL_DIR / "lgbm_fraud.pkl", "rb") as f:
        bundle = pickle.load(f)
    _model = bundle["model"]
    _encoders = bundle.get("encoders", {}) # store encodings for reproducibility
    with open(MODEL_DIR / "feature_cols.json") as f:
        _feature_cols = json.load(f)
    with open(MODEL_DIR / "threshold.json") as f:
        _threshold = json.load(f)["threshold"]


def _get_connection():
    import pyodbc
    from config import SQL_DATABASE, SQL_DRIVER, SQL_SERVER, SQL_TRUST_CERT, USE_WINDOWS_AUTH
    if not USE_WINDOWS_AUTH:
        from config import SQL_USER, SQL_PASSWORD
    trust = "TrustServerCertificate=yes;" if SQL_TRUST_CERT else ""
    if USE_WINDOWS_AUTH:
        conn_str = (
            f"DRIVER={{{SQL_DRIVER}}};"
            f"SERVER={SQL_SERVER};DATABASE={SQL_DATABASE};"
            f"Trusted_Connection=yes;{trust}"
        )
    else:
        conn_str = (
            f"DRIVER={{{SQL_DRIVER}}};"
            f"SERVER={SQL_SERVER};DATABASE={SQL_DATABASE};"
            f"UID={SQL_USER};PWD={SQL_PASSWORD};{trust}"
        )
    return pyodbc.connect(conn_str, autocommit=True)


def _encode_row(row: dict) -> dict:
    """Apply label encoding."""
    encoded = dict(row)
    for col, cats in _encoders.items():
        cat_map = {v: i for i, v in enumerate(cats)}
        val = encoded.get(col, None)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            val = "__NA__"
        encoded[col] = cat_map.get(str(val), -1)
    return encoded


def _build_feature_vector(txn: dict) -> pd.DataFrame:
    encoded = _encode_row(txn)
    row = {}
    for col in _feature_cols:
        v = encoded.get(col, np.nan)
        if isinstance(v, str):
            # fallback for any remaining strings
            v = abs(hash(v)) % 10000
        row[col] = v
    df = pd.DataFrame([row])[_feature_cols]
    # Force columns to float type
    return df.apply(pd.to_numeric, errors="coerce").astype(float)


def _store_decision(result: dict):
    try:
        conn = _get_connection()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO fraud_decisions
                (trans_num, cc_num, merchant, unix_time, amt, fraud_score, decision, is_fraud_actual)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            result["trans_num"],
            result["cc_num"],
            result["merchant"],
            result["unix_time"],
            result["amt"],
            result["fraud_score"],
            result["decision"],
            result["is_fraud_actual"],
        )
        conn.close()
    except Exception as exc:
        print(f"[WARN] SQL write failed: {exc}")


# API

def score_transaction(txn: dict, write_to_sql: bool = True) -> dict:
    """
    Score a single transaction.

    Parameters: 
    txn:
        Must contain all feature columns from models/feature_cols.json.
        Expects 'trans_num', 'cc_num', 'merchant', 'unix_time', 'amt'.
    write_to_sql:
        If True, persist decision to fraud_decisions.

    Returns:
    dict with trans_num, fraud_score, decision, scored_at
    """
    _load_artifacts()

    X = _build_feature_vector(txn)
    score = float(_model.predict_proba(X)[0, 1])

    if score >= _threshold:
        decision = "DECLINE"
    elif score >= REVIEW_THRESHOLD:
        decision = "REVIEW"
    else:
        decision = "APPROVE"

    result = {
        "trans_num": txn.get("trans_num", ""),
        "cc_num": int(txn.get("cc_num", 0)),
        "merchant": str(txn.get("merchant", "")),
        "unix_time": int(txn.get("unix_time", 0)),
        "amt": float(txn.get("amt", 0.0)),
        "fraud_score": round(score, 6),
        "decision": decision,
        "is_fraud_actual": int(txn.get("is_fraud", -1)),
        "scored_at": datetime.utcnow().isoformat()
    }

    if write_to_sql:
        _store_decision(result)

    return result
