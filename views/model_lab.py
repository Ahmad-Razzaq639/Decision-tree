import io
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.tree import export_text, plot_tree

from core import charts
from core.data import pretty
from core.model import (METRIC_COLS, PRESETS, cv_scores, depth_sweep, fit_from_cfg,
                        metrics_at, preset_to_cfg, sk_params)
from core.state import apply_cfg, get_cfg
from core.ui import download_button, header, show_df, show_plot, show_pyplot


def _on_preset():
    ss = st.session_state
    pick = ss.get("lab_preset_pick", "—")
    if pick != "—":
        apply_cfg(preset_to_cfg(PRESETS[pick], keep=ss["cfg"]))
        ss["lab_preset_pick"] = "—"


def _controls(cfg: dict) -> dict:
    ss = st.session_state
    defaults = {
        "w_criterion": cfg["criterion"],
        "w_unlimited": cfg["max_depth"] is None,
        "w_depth": cfg["max_depth"] or 6,
        "w_split": cfg["min_samples_split"],
        "w_leaf": cfg["min_samples_leaf"],
        "w_cw": "balanced" if cfg["class_weight"] == "balanced" else "None",
        "w_ccp": float(cfg["ccp_alpha"]),
        "w_thr": float(cfg["threshold"]),
        "w_test": int(round(cfg["test_size"] * 100)),
        "w_seed": int(cfg["seed"]),
    }
    for k, v in defaults.items():
        ss.setdefault(k, v)

    with st.container(border=True):
        st.markdown("##### ⚙️ Hyperparameters")
        st.selectbox("Load a preset", ["—"] + list(PRESETS), key="lab_preset_pick",
                     on_change=_on_preset)

        criterion = st.selectbox("criterion", ["gini", "entropy"], key="w_criterion",
                                 help="Split quality ka measure. Concepts page par detail hai.")
        unlimited = st.checkbox("Unlimited depth (no max_depth)", key="w_unlimited")
        depth = st.slider("max_depth", 1, 30, key="w_depth", disabled=unlimited,
                          help="Chhota depth = simple tree, overfitting kam.")
        split = st.slider("min_samples_split", 2, 200, key="w_split")
        leaf = st.slider("min_samples_leaf", 1, 100, key="w_leaf")
        cw = st.selectbox("class_weight", ["None", "balanced"], key="w_cw",
                          help="'balanced' churn class ko zyada weight deta hai (recall badhta hai).")
        ccp = st.slider("ccp_alpha (pruning)", 0.0, 0.02, step=0.0005, format="%.4f", key="w_ccp")
        st.markdown("##### 🎯 Evaluation")
        thr = st.slider("Decision threshold", 0.05, 0.95, step=0.01, key="w_thr",
                        help="Churn predict karo agar probability is se zyada ho.")
        test = st.slider("Test size (%)", 10, 40, step=5, key="w_test")
        seed = st.number_input("random_state", 0, 9999, key="w_seed")
        if st.button("↺ Reset to defaults"):
            from core.model import DEFAULT_CFG
            apply_cfg(DEFAULT_CFG)
            st.rerun()

    new = dict(criterion=criterion, max_depth=None if unlimited else int(depth),
               min_samples_split=int(split), min_samples_leaf=int(leaf),
               class_weight=None if cw == "None" else "balanced", ccp_alpha=float(ccp),
               threshold=float(thr), test_size=test / 100, seed=int(seed))
    st.session_state["cfg"] = new
    return new


def render():
    header("🧪 Model Lab", "Decision Tree ke parameters badlo, aur performance live dekho.")
    cfg = get_cfg()
    left, right = st.columns([1, 3], gap="large")
    with left:
        cfg = _controls(cfg)

    res = fit_from_cfg(cfg)
    model, thr = res["model"], cfg["threshold"]
    mt = metrics_at(res["y_test"], res["prob_test"], thr)
    mtr = metrics_at(res["y_train"], res["prob_train"], thr)

    with right:
        cols = st.columns(5)
        for c, name in zip(cols, METRIC_COLS):
            c.metric(name, f"{mt[name]:.3f}",
                     f"train {mtr[name]:.3f}" if name != "ROC-AUC" else f"train {mtr[name]:.3f}",
                     delta_color="off")
        gap = mtr["Accuracy"] - mt["Accuracy"]
        if gap > 0.10:
            st.warning(f"⚠️ Overfitting: train accuracy test se **{gap*100:.1f} pp** zyada hai. "
                       "max_depth ghatao ya min_samples_leaf badhao.")
        elif gap < 0.03:
            st.success("✅ Train aur test accuracy qareeb hain, tree achhi tarah generalize kar raha hai.")
        else:
            st.info(f"Train–test accuracy gap: {gap*100:.1f} pp. Theek hai, thoda tuning se behtar ho sakta hai.")
        st.caption(f"Tree depth **{model.get_depth()}**, leaves **{model.get_n_leaves()}**, "
                   f"train rows **{len(res['y_train']):,}**, test rows **{len(res['y_test']):,}**.")

        tabs = st.tabs(["📊 Performance", "📈 Curves", "🌳 Tree", "⭐ Importance",
                        "🔍 Overfitting", "✅ Cross-validation", "💾 Export"])

        with tabs[0]:
            a, b = st.columns([1, 1.2])
            pred = (res["prob_test"] > thr).astype(int)
            cm = confusion_matrix(res["y_test"], pred)
            with a:
                show_plot(charts.confusion_fig(cm))
            with b:
                p, r, f, s = precision_recall_fscore_support(res["y_test"], pred, zero_division=0)
                rep = pd.DataFrame({"Precision": p, "Recall": r, "F1": f, "Support": s},
                                   index=["No churn (0)", "Churn (1)"])
                st.markdown("**Classification report (test)**")
                show_df(rep.round(3))
                tn, fp, fn, tp = cm.ravel()
                st.markdown(
                    f"- ✅ **{tp}** churners sahi pakde, ❌ **{fn}** churners miss hue\n"
                    f"- ✅ **{tn}** loyal customers sahi, ⚠️ **{fp}** ko galti se churner kaha")

        with tabs[1]:
            a, b = st.columns(2)
            with a:
                show_plot(charts.roc_fig({"Current model": (res["y_test"], res["prob_test"])}))
            with b:
                show_plot(charts.pr_fig(res["y_test"], res["prob_test"]))
            show_plot(charts.threshold_fig(res["y_test"], res["prob_test"], thr))

        with tabs[2]:
            show_depth = st.slider("Levels to display", 1, 5, 3)
            fig, ax = plt.subplots(figsize=(8 + 5 * show_depth, 3.2 + 1.5 * show_depth))
            plot_tree(model, max_depth=show_depth, feature_names=[pretty(f) for f in res["features"]],
                      class_names=["Stay", "Churn"], filled=True, rounded=True, fontsize=8, ax=ax)
            show_pyplot(fig)
            plt.close(fig)
            rules = export_text(model, feature_names=[pretty(f) for f in res["features"]],
                                max_depth=6)
            with st.expander("Text rules (first 6 levels)"):
                st.code(rules, language="text")
            download_button("⬇️ Download rules (.txt)", rules.encode(), "tree_rules.txt", "text/plain")

        with tabs[3]:
            imp = pd.Series(model.feature_importances_, index=res["features"])
            a, b = st.columns([1.4, 1])
            with a:
                show_plot(charts.importance_fig(imp))
            with b:
                tbl = (imp.sort_values(ascending=False).rename("Importance").to_frame()
                       .assign(Feature=lambda d: [pretty(i) for i in d.index]).reset_index(drop=True))
                show_df(tbl[["Feature", "Importance"]].round(4), hide_index=True, height=440)

        with tabs[4]:
            st.caption("Baaki parameters same rakh kar sirf max_depth 1–20 tak badla gaya hai.")
            sw = depth_sweep(sk_params(cfg), cfg["test_size"], cfg["seed"])
            a, b = st.columns(2)
            with a:
                show_plot(charts.depth_fig(sw))
            with b:
                show_plot(charts.depth_fig(sw, ("Test F1", "Test Recall"), "max_depth vs F1 / Recall"))
            best = sw.loc[sw["Test F1"].idxmax()]
            st.info(f"Is setup mein sabse achha **test F1** max_depth = **{int(best['max_depth'])}** par hai "
                    f"(F1 {best['Test F1']:.3f}).")
            show_df(sw.round(4), hide_index=True)

        with tabs[5]:
            cv = cv_scores(sk_params(cfg), cfg["test_size"], cfg["seed"])
            summ = pd.DataFrame({"Mean": cv.mean(), "Std": cv.std()}).round(4)
            a, b = st.columns([1, 1.2])
            with a:
                st.markdown("**5-fold Stratified CV (train set)**")
                show_df(summ)
                show_df(cv.round(4))
            with b:
                import plotly.graph_objects as go
                fig = go.Figure(go.Bar(x=summ.index, y=summ["Mean"], error_y=dict(array=summ["Std"]),
                                       marker_color=charts.C_NO))
                fig.update_layout(title="CV mean ± std", yaxis_range=[0, 1])
                show_plot(charts._fin(fig))

        with tabs[6]:
            buf = io.BytesIO()
            joblib.dump({"model": model, "features": res["features"], "config": cfg}, buf)
            download_button("⬇️ Download trained model (.joblib)", buf.getvalue(),
                            "decision_tree_churn.joblib", "application/octet-stream")
            download_button("⬇️ Download config (.json)", json.dumps(cfg, indent=2).encode(),
                            "model_config.json", "application/json")
            st.code(json.dumps(cfg, indent=2), language="json")
