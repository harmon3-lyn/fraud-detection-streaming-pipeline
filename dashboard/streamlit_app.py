"""
Live fraud monitoring dashboard. (Streamlit)
"""

import sys
import time
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))


st.set_page_config(
    page_title="Fraud Detection Monitor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

REFRESH_SECS = st.sidebar.slider("Refresh interval (s)", 2, 30, 5)
LOOKBACK_N = st.sidebar.number_input("Recent decisions to show", 50, 5000, 500, step=50)

DECISION_COLORS = {
    "APPROVE": "#2ecc71",
    "REVIEW": "#f39c12",
    "DECLINE": "#e74c3c",
}


# SQL Setup

def get_engine():
    from sqlalchemy import create_engine
    from sqlalchemy.pool import NullPool
    from config import (SQL_DATABASE, SQL_DRIVER, SQL_SERVER, SQL_TRUST_CERT, USE_WINDOWS_AUTH)

    if not USE_WINDOWS_AUTH:
        from config import SQL_USER, SQL_PASSWORD
    driver = SQL_DRIVER.replace(" ", "+")
    trust  = "&TrustServerCertificate=yes" if SQL_TRUST_CERT else ""

    if USE_WINDOWS_AUTH:
        conn_str = (
            f"mssql+pyodbc://@{SQL_SERVER}/{SQL_DATABASE}"
            f"?driver={driver}&trusted_connection=yes{trust}"
        )
    else:
        conn_str = (
            f"mssql+pyodbc://{SQL_USER}:{SQL_PASSWORD}@{SQL_SERVER}/{SQL_DATABASE}"
            f"?driver={driver}{trust}"
        )
    return create_engine(conn_str, poolclass=NullPool)



def fetch_decisions(n: int) -> pd.DataFrame:
    engine = get_engine()
    query = f"""
        SELECT TOP {n}
            id, 
            trans_num, 
            cc_num, 
            merchant, 
            amt,
            fraud_score, 
            decision, 
            is_fraud_actual, 
            scored_at
        FROM fraud_decisions
        ORDER BY scored_at DESC
    """
    with engine.connect() as conn:
        return pd.read_sql(query, conn)


def fetch_hourly_metrics() -> pd.DataFrame:
    engine = get_engine()
    query = """
        SELECT *
        FROM monitoring_metrics
        ORDER BY decision_date, decision_hour
    """
    with engine.connect() as conn:
        return pd.read_sql(query, conn)


# Main Dashboard

placeholder = st.empty()

try:
    df = fetch_decisions(int(LOOKBACK_N))
    metrics = fetch_hourly_metrics()
except Exception as exc:
    st.error(f"DB connection error: {exc}")
    time.sleep(REFRESH_SECS)
    st.rerun()

with placeholder.container():
        st.title(" Fraud Detection - Live Monitor")
        st.caption(f"Last refresh: {pd.Timestamp.now().strftime('%H:%M:%S')}  |  "
                   f"Showing last {len(df)} decisions")

        # top-level KPIs
        if df.empty:
            st.info("No decisions yet. Start the simulator or API.")
        else:
            total = len(df)
            n_dec = (df["decision"] == "DECLINE").sum()
            n_rev = (df["decision"] == "REVIEW").sum()
            n_app = (df["decision"] == "APPROVE").sum()
            n_fraud = (df["is_fraud_actual"] == 1).sum()
            fraud_rate = n_fraud / total * 100
            decline_rate = n_dec / total * 100
            avg_score = df["fraud_score"].mean()
            blocked_amt = df.loc[df["decision"] == "DECLINE", "amt"].sum()

            k1, k2, k3, k4, k5, k6 = st.columns(6)
            k1.metric("Total Transactions", f"{total:,}")
            k2.metric("Declined", f"{n_dec:,}", f"{decline_rate:.1f}%")
            k3.metric("Review", f"{n_rev:,}")
            k4.metric("Approved", f"{n_app:,}")
            k5.metric("Fraud Rate (actual)", f"{fraud_rate:.2f}%")
            k6.metric("Blocked Amount", f"${blocked_amt:,.0f}")

            st.divider()


            # row 1: decision pie chart + score distribution
            col_pie, col_hist = st.columns(2)

            with col_pie:
                st.subheader("Decision Breakdown")
                pie_data = df["decision"].value_counts().reset_index()
                pie_data.columns = ["Decision", "Count"]
                fig_pie = px.pie(
                    pie_data, values="Count", names="Decision",
                    color="Decision",
                    color_discrete_map=DECISION_COLORS,
                    hole=0.4,
                )
                fig_pie.update_traces(
                    textposition="inside",
                    textinfo="percent+label",
                    insidetextorientation="radial",
                )
                fig_pie.update_layout(
                    margin=dict(t=10, b=10, l=10, r=10),
                    height=280,
                    showlegend=False,
                )
                st.plotly_chart(fig_pie, use_container_width=True, key="fig_pie")

            with col_hist:
                st.subheader("Score Distribution")
                fig_hist = px.histogram(
                    df, x="fraud_score", nbins=50,
                    color="decision",
                    color_discrete_map=DECISION_COLORS,
                    barmode="overlay",
                    opacity=0.75,
                )
                fig_hist.update_layout(
                    margin=dict(t=10, b=10, l=10, r=10), height=280,
                    xaxis_title="Fraud Score", yaxis_title="Count",
                    legend_title="Decision",
                )
                st.plotly_chart(fig_hist, use_container_width=True, key="fig_hist")


            # row 2: hourly volume
            if not metrics.empty:
                st.subheader("Hourly Volume")
                metrics["hour_label"] = (metrics["decision_date"].astype(str) + " " + metrics["decision_hour"].astype(str).str.zfill(2) + ":00")
                fig_vol = go.Figure()
                for decision, color in DECISION_COLORS.items():
                    col = decision.lower() + "d" if decision == "DECLINE" else decision.lower() + "d"
                    # map column names from view
                    col_map = {"APPROVE": "approved", "REVIEW": "reviewed", "DECLINE": "declined"}
                    fig_vol.add_bar(
                        x=metrics["hour_label"],
                        y=metrics[col_map[decision]],
                        name=decision,
                        marker_color=color,
                    )
                fig_vol.update_layout(
                    barmode="stack",
                    height=260,
                    margin=dict(t=10, b=10, l=10, r=10),
                    xaxis_title="Hour",
                    yaxis_title="Transactions",
                    legend_title="Decision",
                )
                st.plotly_chart(fig_vol, use_container_width=True, key="fig_vol")


            # row 3: recent decisions (table)
            st.subheader("Recent Decisions")

            display = df.copy()
            display["scored_at"] = pd.to_datetime(display["scored_at"]).dt.strftime("%H:%M:%S")
            display["fraud_score"] = display["fraud_score"].map("{:.6f}".format)
            display["amt"] = display["amt"].map("${:,.2f}".format)

            def color_row(row):
                c = DECISION_COLORS.get(row["decision"], "")
                alpha = "33" 
                return [f"background-color: {c}{alpha}"] * len(row)

            styled = (
                display[["scored_at", "trans_num", "merchant", "amt", "fraud_score", "decision", "is_fraud_actual"]]
                .rename(columns={"is_fraud_actual": "actual_fraud", "scored_at": "time"})
                .style.apply(color_row, axis=1)
            )
            st.dataframe(styled, use_container_width=True, height=400)

time.sleep(REFRESH_SECS)
st.rerun()
