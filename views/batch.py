import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from core import charts
from core.data import RAW_FEATURES, get_eda, load_raw, preprocess, total_charges_median, validate_raw
from core.model import fit_from_cfg, metrics_at, METRIC_COLS
from core.state import get_cfg
from core.ui import download_button, header, risk_label, show_df, show_plot


def render():
    header("📦 Batch Prediction", "Poori CSV upload karo, har customer ka churn risk nikalo, aur results download karo.")
    cfg = get_cfg()
    res = fit_from_cfg(cfg)
    model, features, thr = res["model"], res["features"], cfg["threshold"]

    st.caption("CSV ke columns Telco dataset jaise hone chahiye (customerID optional, Churn optional).")
    template = load_raw().drop(columns=["Churn"]).sample(10, random_state=1)
    a, b = st.columns([1, 3])
    with a:
        download_button("⬇️ Download sample template", template.to_csv(index=False).encode(),
                        "churn_template.csv", "text/csv")
    with b:
        use_demo = st.toggle("Use 300 random rows from the dataset as a demo (with actual Churn)")

    up = st.file_uploader("Upload CSV", type=["csv"])
    if use_demo:
        df = load_raw().sample(300, random_state=7).reset_index(drop=True)
    elif up is not None:
        try:
            df = pd.read_csv(up)
        except Exception as e:  # noqa: BLE001
            st.error(f"CSV read nahi hui: {e}")
            return
    else:
        st.info("⬆️ CSV upload karo ya demo toggle on karo.")
        return

    missing, bad = validate_raw(df)
    if missing:
        st.error(f"Ye columns missing hain: {', '.join(missing)}")
        return
    if bad:
        st.error("Kuch columns mein unexpected values hain:")
        show_df(pd.DataFrame({"Column": list(bad), "Unexpected values": [", ".join(v[:6]) for v in bad.values()]}),
                hide_index=True)
        return

    X = preprocess(df, total_charges_median()).reindex(columns=features, fill_value=0)
    prob = model.predict_proba(X)[:, 1]
    out = df.copy()
    out["churn_probability"] = np.round(prob, 4)
    out["risk_level"] = [risk_label(p) for p in prob]
    out["predicted_churn"] = np.where(prob > thr, "Yes", "No")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Customers scored", f"{len(out):,}")
    k2.metric("Predicted to churn", f"{(prob > thr).sum():,}", f"{(prob > thr).mean()*100:.1f}%", delta_color="off")
    k3.metric("High-risk (≥60%)", f"{(prob >= .6).sum():,}")
    k4.metric("Avg churn probability", f"{prob.mean()*100:.1f}%")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(out, x="churn_probability", nbins=25, title="Distribution of churn probability",
                           color_discrete_sequence=[charts.C_YES])
        show_plot(charts._fin(fig))
    with c2:
        cnt = out["risk_level"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0)
        fig = px.bar(x=cnt.index, y=cnt.values, color=cnt.index,
                     color_discrete_map={"Low": "#2FA36B", "Medium": "#E8A33C", "High": "#D64545"},
                     title="Customers per risk level", labels={"x": "", "y": "Customers"})
        fig.update_layout(showlegend=False)
        show_plot(charts._fin(fig))

    if "Churn" in df.columns:
        y = df["Churn"].map({"No": 0, "Yes": 1})
        if y.notna().all():
            st.markdown("**Actual Churn column mila: is batch par model ki performance**")
            m = metrics_at(y, prob, thr)
            cols = st.columns(5)
            for c, name in zip(cols, METRIC_COLS):
                c.metric(name, f"{m[name]:.3f}")
            st.caption("Dhyan: agar ye rows training data ka hissa hain to scores optimistic honge.")

    st.markdown("#### Results")
    risk = st.multiselect("Show risk levels", ["High", "Medium", "Low"], default=["High", "Medium"])
    view = out[out["risk_level"].isin(risk)].sort_values("churn_probability", ascending=False)
    show_df(view, hide_index=True, height=420,
            column_config={"churn_probability": st.column_config.ProgressColumn(
                "churn_probability", min_value=0, max_value=1, format="%.2f")})
    download_button("⬇️ Download all predictions (CSV)", out.to_csv(index=False).encode(),
                    "churn_predictions.csv", "text/csv", type="primary")
