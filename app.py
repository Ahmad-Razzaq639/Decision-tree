"""Telco Customer Churn Studio - Streamlit app (Decision Tree)."""
import streamlit as st

st.set_page_config(page_title="Telco Churn Studio", page_icon="📉", layout="wide",
                   initial_sidebar_state="expanded")

from core.state import get_cfg  # noqa: E402
from core.model import fit_from_cfg, metrics_at  # noqa: E402
from core.ui import inject_css  # noqa: E402
from views import batch, compare, concepts, explorer, model_lab, overview, predict  # noqa: E402

PAGES = {
    "📊 Overview": overview.render,
    "🗂️ Data Explorer": explorer.render,
    "🧪 Model Lab": model_lab.render,
    "⚖️ Compare Models": compare.render,
    "🔮 Predict": predict.render,
    "📦 Batch Predict": batch.render,
    "📚 Concepts": concepts.render,
}

inject_css()

with st.sidebar:
    st.markdown("## 📉 Telco Churn Studio")
    st.caption("Decision Tree · scikit-learn · Streamlit")
    page = st.radio("Navigate", list(PAGES), label_visibility="collapsed", key="nav")
    st.divider()

    cfg = get_cfg()
    res = fit_from_cfg(cfg)
    m = metrics_at(res["y_test"], res["prob_test"], cfg["threshold"])
    st.markdown("**Active model**")
    st.caption(f"{cfg['criterion']} · depth {cfg['max_depth']} · leaf {cfg['min_samples_leaf']} · "
               f"split {cfg['min_samples_split']} · cw {cfg['class_weight']} · thr {cfg['threshold']:.2f}")
    c1, c2 = st.columns(2)
    c1.metric("F1", f"{m['F1']:.3f}")
    c2.metric("Recall", f"{m['Recall']:.3f}")
    c1.metric("ROC-AUC", f"{m['ROC-AUC']:.3f}")
    c2.metric("Accuracy", f"{m['Accuracy']:.3f}")
    st.caption("Model Lab mein parameters badlo, saare pages par apply honge.")

PAGES[page]()
