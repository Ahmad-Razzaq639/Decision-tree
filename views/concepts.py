import numpy as np
import streamlit as st

from core import charts
from core.ui import header, show_df, show_plot
import pandas as pd


def gini(*counts):
    n = sum(counts)
    return 0.0 if n == 0 else 1 - sum((c / n) ** 2 for c in counts)


def entropy(*counts):
    n = sum(counts)
    if n == 0:
        return 0.0
    return -sum((c / n) * np.log2(c / n) for c in counts if c > 0)


def render():
    header("📚 Concepts & Pipeline", "Gini, Entropy, Information Gain aur poora preprocessing flow.")
    t1, t2, t3, t4 = st.tabs(["Impurity calculator", "Split & Information Gain", "Gini vs Entropy", "Pipeline & data dictionary"])

    with t1:
        a, b = st.columns(2)
        no = a.number_input("Stay (No churn) customers in node", 0, 100000, 73)
        yes = b.number_input("Churn customers in node", 0, 100000, 27)
        g, e = gini(no, yes), entropy(no, yes)
        c1, c2 = st.columns(2)
        c1.metric("Gini index", f"{g:.4f}", help="1 − Σ pᵢ²   (0 = pure, 0.5 = 50/50)")
        c2.metric("Entropy", f"{e:.4f}", help="−Σ pᵢ·log₂(pᵢ)   (0 = pure, 1 = 50/50)")
        st.latex(r"Gini = 1-\sum p_i^2 \qquad Entropy=-\sum p_i\log_2 p_i")

    with t2:
        st.markdown("Ek parent node ko do child nodes mein todo aur dekho split kitna faida deta hai.")
        a, b, c = st.columns(3)
        with a:
            st.markdown("**Parent**")
            pn = st.number_input("Stay", 0, 100000, 1000, key="pn")
            py = st.number_input("Churn", 0, 100000, 400, key="py")
        with b:
            st.markdown("**Left child**")
            ln = st.number_input("Stay", 0, pn, min(300, pn), key="ln")
            ly = st.number_input("Churn", 0, py, min(50, py), key="ly")
        rn, ry = pn - ln, py - ly
        with c:
            st.markdown("**Right child (auto)**")
            st.metric("Stay", rn)
            st.metric("Churn", ry)
        n, nl, nr = pn + py, ln + ly, rn + ry
        if n == 0 or nl == 0 or nr == 0:
            st.warning("Dono children mein kam az kam ek customer hona chahiye.")
        else:
            ig = entropy(pn, py) - (nl / n) * entropy(ln, ly) - (nr / n) * entropy(rn, ry)
            gd = gini(pn, py) - (nl / n) * gini(ln, ly) - (nr / n) * gini(rn, ry)
            tbl = pd.DataFrame({
                "Node": ["Parent", "Left", "Right", "Weighted children"],
                "Samples": [n, nl, nr, n],
                "Gini": [gini(pn, py), gini(ln, ly), gini(rn, ry),
                         (nl / n) * gini(ln, ly) + (nr / n) * gini(rn, ry)],
                "Entropy": [entropy(pn, py), entropy(ln, ly), entropy(rn, ry),
                            (nl / n) * entropy(ln, ly) + (nr / n) * entropy(rn, ry)]})
            show_df(tbl.round(4), hide_index=True)
            m1, m2 = st.columns(2)
            m1.metric("Information Gain (entropy)", f"{ig:.4f}")
            m2.metric("Gini decrease", f"{gd:.4f}")
            st.latex(r"IG = H(parent) - \sum_{k}\frac{n_k}{n}\,H(child_k)")
            st.caption("Tree har node par wohi split chunta hai jis mein ye gain sabse zyada ho.")

    with t3:
        show_plot(charts.impurity_curves())
        st.markdown("- Dono 50/50 mix par sabse zyada aur pure node par 0 hote hain.\n"
                    "- Gini thoda tez compute hota hai (log nahi), isliye sklearn ka default hai.\n"
                    "- Practical results aksar milte julte hote hain. Compare page par dono ko test karo.")

    with t4:
        st.markdown("""
**Preprocessing (notebook ke mutabiq)**

| Column(s) | Treatment |
|---|---|
| `customerID` | Drop |
| `gender` | Female=0, Male=1 |
| `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling` | No=0, Yes=1 |
| `MultipleLines` | "No phone service" → No, phir binary |
| `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` | "No internet service" → No, phir binary |
| `InternetService`, `Contract`, `PaymentMethod` | One-hot encoding |
| `TotalCharges` | Numeric, 11 blank values median se fill |
| `Churn` | No=0, Yes=1 (target) |

**Split:** 80/20 stratified, `random_state=42`.

**Scaling:** Notebook mein StandardScaler fit hua tha. Decision Tree monotonic transformations se
invariant hai, isliye app mein scaling skip ki gayi hai. Tuned trees ke predictions notebook jaise hi rehte hain,
aur thresholds asli units mein (jaise tenure ≤ 17.5 months) nazar aati hain. Sirf bohat gehre default tree
mein 1–2 barabar-gain splits ka tie-break badalne se 0.1% ke qareeb farq aa sakta hai.

**Imbalance:** ~73% stay / ~27% churn, isliye accuracy ke saath Recall, F1 aur ROC-AUC dekhna zaroori hai.
""")
