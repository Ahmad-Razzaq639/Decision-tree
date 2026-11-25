import streamlit as st

from core import charts
from core.model import (BASE_SK, METRIC_COLS, PRESETS, GRID, depth_sweep, fit_from_cfg,
                        fit_model, metrics_at, preset_table, preset_to_cfg, run_grid_search)
from core.state import apply_cfg, get_cfg
from core.ui import button, header, show_df, show_plot
import pandas as pd


def _colcfg():
    cfg = {m: st.column_config.ProgressColumn(m, min_value=0, max_value=1, format="%.3f")
           for m in ["Train Acc", "Test Acc", "Precision", "Recall", "F1", "ROC-AUC"]}
    return cfg


def render():
    header("⚖️ Model Comparison",
           "Notebook ke saare configurations, Gini vs Entropy, aur GridSearchCV ek jagah.")
    cfg = get_cfg()
    ts, seed = cfg["test_size"], cfg["seed"]

    tab1, tab2, tab3 = st.tabs(["Preset configurations", "Gini vs Entropy", "GridSearchCV"])

    with tab1:
        tbl = preset_table(ts, seed).copy()
        cur = fit_from_cfg(cfg)
        m = metrics_at(cur["y_test"], cur["prob_test"], 0.5)
        tbl.loc[len(tbl)] = {
            "Model": "★ Current Model Lab config",
            "Train Acc": metrics_at(cur["y_train"], cur["prob_train"], 0.5)["Accuracy"],
            "Test Acc": m["Accuracy"], "Precision": m["Precision"], "Recall": m["Recall"],
            "F1": m["F1"], "ROC-AUC": m["ROC-AUC"],
            "Depth": cur["model"].get_depth(), "Leaves": cur["model"].get_n_leaves()}

        best_f1 = tbl.loc[tbl["F1"].idxmax()]
        best_rc = tbl.loc[tbl["Recall"].idxmax()]
        best_auc = tbl.loc[tbl["ROC-AUC"].idxmax()]
        a, b, c = st.columns(3)
        a.metric("Best F1", f"{best_f1['F1']:.3f}", best_f1["Model"], delta_color="off")
        b.metric("Best Recall", f"{best_rc['Recall']:.3f}", best_rc["Model"], delta_color="off")
        c.metric("Best ROC-AUC", f"{best_auc['ROC-AUC']:.3f}", best_auc["Model"], delta_color="off")

        show_df(tbl.round(4), hide_index=True, column_config=_colcfg())
        st.caption("Threshold 0.5 par compare kiya gaya hai (notebook jaisa). Tuned configs ke numbers notebook se bilkul match karte hain. "
                   "Sirf fully-grown Basic tree mein chhota sa farq (~0.1–0.8%) ho sakta hai, kyunke app mein scaling skip hai "
                   "aur deep tree mein barabar-gain splits ka tie-break thoda badal jata hai.")

        sel = st.multiselect("Metrics in chart", METRIC_COLS, default=["Precision", "Recall", "F1", "ROC-AUC"])
        if sel:
            show_plot(charts.compare_bars(tbl.rename(columns={}), sel))

        curves = {}
        for name, p in PRESETS.items():
            r = fit_model({**BASE_SK, **p}, ts, seed)
            curves[name] = (r["y_test"], r["prob_test"])
        curves["★ Current"] = (cur["y_test"], cur["prob_test"])
        show_plot(charts.roc_fig(curves))

    with tab2:
        st.caption("Baaki parameters Model Lab wale rakhe gaye hain, sirf criterion alag hai.")
        base = {k: cfg[k] for k in BASE_SK}
        g = depth_sweep({**base, "criterion": "gini"}, ts, seed).assign(criterion="gini")
        e = depth_sweep({**base, "criterion": "entropy"}, ts, seed).assign(criterion="entropy")
        merged = pd.DataFrame({"max_depth": g["max_depth"],
                               "Gini · Test F1": g["Test F1"], "Entropy · Test F1": e["Test F1"],
                               "Gini · Test Acc": g["Test Acc"], "Entropy · Test Acc": e["Test Acc"]})
        a, b = st.columns(2)
        with a:
            show_plot(charts.depth_fig(merged, ("Gini · Test F1", "Entropy · Test F1"), "Test F1: Gini vs Entropy"))
        with b:
            show_plot(charts.depth_fig(merged, ("Gini · Test Acc", "Entropy · Test Acc"), "Test accuracy: Gini vs Entropy"))
        wins = int((merged["Gini · Test F1"] > merged["Entropy · Test F1"]).sum())
        st.info(f"20 depths mein se Gini **{wins}** par aur Entropy **{int((merged['Entropy · Test F1'] > merged['Gini · Test F1']).sum())}** "
                "par F1 mein aage hai. Aksar farq chhota hota hai.")
        show_df(merged.round(4), hide_index=True)

    with tab3:
        n_combo = 1
        for v in GRID.values():
            n_combo *= len(v)
        st.markdown(f"Grid: **{n_combo}** combinations × 5 folds = **{n_combo*5}** fits.")
        st.json({k: [str(x) for x in v] for k, v in GRID.items()}, expanded=False)
        scoring = st.selectbox("Optimise for", ["f1", "recall", "roc_auc", "accuracy", "precision"])
        if button("🚀 Run GridSearchCV", type="primary"):
            with st.spinner("Searching best hyperparameters..."):
                st.session_state["grid_res"] = (scoring, run_grid_search(ts, seed, scoring))
        if "grid_res" in st.session_state:
            sc, res = st.session_state["grid_res"]
            st.success(f"Best CV {sc}: **{res['best_score']:.4f}** ({res['n_fits']} fits)")
            st.json(res["best_params"])
            tm = res["test_metrics"]
            cols = st.columns(5)
            for c, name in zip(cols, METRIC_COLS):
                c.metric(f"Test {name}", f"{tm[name]:.3f}")
            show_df(res["top"].round(4), hide_index=True)

            def _apply(p=res["best_params"]):
                apply_cfg(preset_to_cfg(p, keep=st.session_state["cfg"]))
            st.button("✅ Use these params in Model Lab & Predict", on_click=_apply)
