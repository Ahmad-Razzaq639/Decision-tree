import numpy as np
import pandas as pd
import streamlit as st

from core import charts
from core.data import ADDONS, CATS, get_eda, preprocess, total_charges_median
from core.model import explain_path, fit_from_cfg
from core.state import get_cfg
from core.ui import button, header, risk_badge, show_df, show_plot

ADDON_KEYS = {a: f"p_{a}" for a in ADDONS}

EXAMPLES = {
    "high": dict(gender="Female", SeniorCitizen=0, Partner="No", Dependents="No", tenure=2,
                 PhoneService="Yes", MultipleLines="No", InternetService="Fiber optic",
                 OnlineSecurity="No", OnlineBackup="No", DeviceProtection="No", TechSupport="No",
                 StreamingTV="No", StreamingMovies="No", Contract="Month-to-month",
                 PaperlessBilling="Yes", PaymentMethod="Electronic check",
                 MonthlyCharges=85.0, TotalCharges=170.0),
    "low": dict(gender="Male", SeniorCitizen=0, Partner="Yes", Dependents="Yes", tenure=60,
                PhoneService="Yes", MultipleLines="Yes", InternetService="DSL",
                OnlineSecurity="Yes", OnlineBackup="Yes", DeviceProtection="Yes", TechSupport="Yes",
                StreamingTV="No", StreamingMovies="No", Contract="Two year",
                PaperlessBilling="No", PaymentMethod="Bank transfer (automatic)",
                MonthlyCharges=65.0, TotalCharges=3900.0),
}


def _set_record(r: dict, auto_total=False):
    ss = st.session_state
    ss["p_gender"] = r["gender"]
    ss["p_senior"] = "Yes" if int(r["SeniorCitizen"]) == 1 else "No"
    ss["p_partner"], ss["p_dep"] = r["Partner"], r["Dependents"]
    ss["p_tenure"] = int(r["tenure"])
    ss["p_phone"] = r["PhoneService"]
    ss["p_lines"] = "Yes" if r["MultipleLines"] == "Yes" else "No"
    ss["p_internet"] = r["InternetService"]
    for a in ADDONS:
        ss[ADDON_KEYS[a]] = "Yes" if r[a] == "Yes" else "No"
    ss["p_contract"], ss["p_paperless"], ss["p_payment"] = r["Contract"], r["PaperlessBilling"], r["PaymentMethod"]
    ss["p_monthly"] = float(r["MonthlyCharges"])
    ss["p_auto"] = auto_total
    total = pd.to_numeric(r["TotalCharges"], errors="coerce")
    ss["p_total"] = float(0.0 if pd.isna(total) else total)


def _load_example(kind):
    _set_record(EXAMPLES[kind])
    st.session_state["p_sample_note"] = None


def _load_random():
    df = get_eda()
    row = df.sample(1).iloc[0].to_dict()
    row["SeniorCitizen"] = 1 if row["SeniorCitizen"] == "Yes" else 0
    _set_record(row)
    st.session_state["p_sample_note"] = (
        f"Real customer **{row['customerID']}**: actual outcome = "
        f"**{'CHURNED' if row['Churn'] == 'Yes' else 'STAYED'}**")


P_KEYS = (["p_gender", "p_senior", "p_partner", "p_dep", "p_tenure", "p_phone", "p_lines",
           "p_internet", "p_contract", "p_paperless", "p_payment", "p_monthly", "p_auto", "p_total"]
          + list(ADDON_KEYS.values()))


def _init_defaults():
    """Streamlit drops widget state when a page isn't rendered; restore it from a snapshot."""
    ss = st.session_state
    if "p_tenure" not in ss:
        if "p_saved" in ss:
            ss.update(ss["p_saved"])
        else:
            _set_record(EXAMPLES["high"], auto_total=True)


def _save_snapshot():
    ss = st.session_state
    ss["p_saved"] = {k: ss[k] for k in P_KEYS if k in ss}


def _record() -> dict:
    ss = st.session_state
    internet, phone = ss["p_internet"], ss["p_phone"]
    monthly, tenure = float(ss["p_monthly"]), int(ss["p_tenure"])
    if ss["p_auto"]:
        total = monthly * tenure if tenure > 0 else np.nan
    else:
        total = float(ss["p_total"])
    rec = dict(
        gender=ss["p_gender"], SeniorCitizen=1 if ss["p_senior"] == "Yes" else 0,
        Partner=ss["p_partner"], Dependents=ss["p_dep"], tenure=tenure,
        PhoneService=phone, MultipleLines="No phone service" if phone == "No" else ss["p_lines"],
        InternetService=internet, PaperlessBilling=ss["p_paperless"],
        Contract=ss["p_contract"], PaymentMethod=ss["p_payment"],
        MonthlyCharges=monthly, TotalCharges=total)
    for a in ADDONS:
        rec[a] = "No internet service" if internet == "No" else ss[ADDON_KEYS[a]]
    return rec


def _prob(model, features, records: list) -> np.ndarray:
    X = preprocess(pd.DataFrame(records), total_charges_median())
    return model.predict_proba(X.reindex(columns=features, fill_value=0))[:, 1]


def _form():
    ss = st.session_state
    b1, b2, b3 = st.columns(3)
    b1.button("🔴 High-risk example", on_click=_load_example, args=("high",))
    b2.button("🟢 Loyal example", on_click=_load_example, args=("low",))
    b3.button("🎲 Random real customer", on_click=_load_random)
    if ss.get("p_sample_note"):
        st.info(ss["p_sample_note"])

    t1, t2, t3, t4 = st.tabs(["👤 Profile", "📝 Account", "🌐 Services", "💵 Charges"])
    with t1:
        a, b = st.columns(2)
        a.selectbox("Gender", ["Female", "Male"], key="p_gender")
        b.selectbox("Senior citizen", ["No", "Yes"], key="p_senior")
        a.selectbox("Has partner", ["No", "Yes"], key="p_partner")
        b.selectbox("Has dependents", ["No", "Yes"], key="p_dep")
    with t2:
        st.slider("Tenure (months)", 0, 72, key="p_tenure")
        a, b = st.columns(2)
        a.selectbox("Contract", CATS["Contract"], key="p_contract")
        b.selectbox("Payment method", CATS["PaymentMethod"], key="p_payment")
        a.selectbox("Paperless billing", ["No", "Yes"], key="p_paperless")
    with t3:
        a, b = st.columns(2)
        a.selectbox("Phone service", ["No", "Yes"], key="p_phone")
        b.selectbox("Multiple lines", ["No", "Yes"], key="p_lines", disabled=ss["p_phone"] == "No")
        st.selectbox("Internet service", CATS["InternetService"], key="p_internet")
        no_net = ss["p_internet"] == "No"
        cols = st.columns(3)
        for i, a_ in enumerate(ADDONS):
            cols[i % 3].selectbox(a_, ["No", "Yes"], key=ADDON_KEYS[a_], disabled=no_net)
        if no_net:
            st.caption("Internet service nahi hai, isliye add-ons automatically 'No internet service' count honge.")
    with t4:
        st.slider("Monthly charges ($)", 18.0, 120.0, step=0.05, key="p_monthly")
        st.checkbox("Total charges = tenure × monthly (auto)", key="p_auto")
        st.number_input("Total charges ($)", 0.0, 10000.0, step=10.0, key="p_total", disabled=ss["p_auto"])


def render():
    header("🔮 Predict Churn", "Ek customer ki details bharo, probability aur wajah dekho, aur what-if try karo.")
    _init_defaults()
    cfg = get_cfg()
    res = fit_from_cfg(cfg)
    model, features, thr = res["model"], res["features"], cfg["threshold"]

    st.caption(f"Model: `{cfg['criterion']}` · depth `{cfg['max_depth']}` · leaf `{cfg['min_samples_leaf']}` · "
               f"split `{cfg['min_samples_split']}` · class_weight `{cfg['class_weight']}` · threshold `{thr}`. "
               "Model Lab mein badlo to yahan bhi apply hoga.")

    left, right = st.columns([1.25, 1], gap="large")
    with left:
        _form()

    _save_snapshot()
    rec = _record()
    prob = float(_prob(model, features, [rec])[0])
    X_row = preprocess(pd.DataFrame([rec]), total_charges_median()).reindex(columns=features, fill_value=0)

    with right:
        with st.container(border=True):
            show_plot(charts.gauge(prob, thr), key="gauge")
            verdict = "WILL LIKELY CHURN" if prob > thr else "LIKELY TO STAY"
            st.markdown(f"<div style='text-align:center'>{risk_badge(prob)}"
                        f"<h3 style='margin:.4rem 0'>{verdict}</h3>"
                        f"<span class='small-note'>Black marker = decision threshold ({thr:.2f})</span></div>",
                        unsafe_allow_html=True)
        path, info = explain_path(model, X_row)
        st.markdown("**🌳 Tree ne ye rules apply kiye**")
        show_df(path, hide_index=True)
        st.caption(f"Is leaf mein training ke **{info['leaf_samples']:,}** customers aaye. "
                   "Probability unhi mein churn karne walon ka (weighted) hissa hai.")

    st.divider()
    tabs = st.tabs(["🎛️ Retention levers", "📑 Contract comparison", "📈 Sensitivity", "🧾 Input record"])

    with tabs[0]:
        base = prob
        monthly = rec["MonthlyCharges"]
        levers = []

        def add(label, **chg):
            if all(rec.get(k) == v for k, v in chg.items()):
                return
            r2 = {**rec, **chg}
            levers.append((label, r2))

        add("Move to 1-year contract", Contract="One year")
        add("Move to 2-year contract", Contract="Two year")
        add("Switch to automatic bank transfer", PaymentMethod="Bank transfer (automatic)")
        add("Switch to automatic credit card", PaymentMethod="Credit card (automatic)")
        if rec["InternetService"] != "No":
            for a_ in ["TechSupport", "OnlineSecurity", "OnlineBackup", "DeviceProtection"]:
                add(f"Add {a_}", **{a_: "Yes"})
        add("Turn paperless billing " + ("off" if rec["PaperlessBilling"] == "Yes" else "on"),
            PaperlessBilling="No" if rec["PaperlessBilling"] == "Yes" else "Yes")
        disc_total = rec["TotalCharges"] * 0.9 if not pd.isna(rec["TotalCharges"]) else np.nan
        levers.append(("Give 10% discount on monthly bill",
                       {**rec, "MonthlyCharges": monthly * 0.9, "TotalCharges": disc_total}))

        probs = _prob(model, features, [r for _, r in levers])
        df = pd.DataFrame({"Action": [l for l, _ in levers],
                           "New churn probability": probs * 100,
                           "Change (pp)": (probs - base) * 100}).sort_values("Change (pp)")
        show_df(df, hide_index=True, column_config={
            "New churn probability": st.column_config.NumberColumn(format="%.1f%%"),
            "Change (pp)": st.column_config.NumberColumn(format="%+.1f pp")})
        best = df.iloc[0]
        if best["Change (pp)"] < -0.5:
            st.success(f"Sabse bara asar: **{best['Action']}** ({best['Change (pp)']:+.1f} pp).")
        else:
            st.info("Is customer ke liye in changes se model ki probability mein khaas farq nahi aata.")
        st.caption("⚠️ Ye model ke patterns hain (correlation), guaranteed causal effect nahi. "
                   "Decision Tree ki probabilities leaf-wise step jaisi hoti hain.")

    with tabs[1]:
        opts = CATS["Contract"]
        ps = _prob(model, features, [{**rec, "Contract": c} for c in opts]) * 100
        show_plot(charts.bar_simple(opts, list(ps), "Churn probability by contract type",
                                    highlight=rec["Contract"]), key="contract_cmp")
        pay = CATS["PaymentMethod"]
        pp = _prob(model, features, [{**rec, "PaymentMethod": c} for c in pay]) * 100
        show_plot(charts.bar_simple(pay, list(pp), "Churn probability by payment method",
                                    highlight=rec["PaymentMethod"]), key="pay_cmp")

    with tabs[2]:
        a, b = st.columns(2)
        auto = st.session_state["p_auto"]
        ten = list(range(0, 73))
        recs = [{**rec, "tenure": t, "TotalCharges": (rec["MonthlyCharges"] * t if t > 0 else np.nan)
                 if auto else rec["TotalCharges"]} for t in ten]
        with a:
            show_plot(charts.line_simple(ten, _prob(model, features, recs) * 100,
                                         "Churn probability vs tenure", "Tenure (months)", rec["tenure"]),
                      key="sens_t")
        mons = list(np.round(np.arange(18, 120.5, 1.0), 2))
        recs = [{**rec, "MonthlyCharges": m_, "TotalCharges": (m_ * rec["tenure"] if rec["tenure"] > 0 else np.nan)
                 if auto else rec["TotalCharges"]} for m_ in mons]
        with b:
            show_plot(charts.line_simple(mons, _prob(model, features, recs) * 100,
                                         "Churn probability vs monthly charges", "Monthly charges ($)",
                                         rec["MonthlyCharges"]), key="sens_m")

    with tabs[3]:
        st.markdown("Raw record jo model ko gaya (encoded features neeche):")
        show_df(pd.DataFrame([rec]).T.rename(columns={0: "value"}).astype(str))
        with st.expander("Encoded features"):
            show_df(X_row.T.rename(columns={X_row.index[0]: "value"}))
