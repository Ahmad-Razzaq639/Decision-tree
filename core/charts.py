"""All Plotly figures in one place."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import precision_recall_curve, roc_curve

from core.data import pretty

C_NO, C_YES = "#4C9BE8", "#E8734C"
CMAP = {"No": C_NO, "Yes": C_YES}
MARGIN = dict(l=10, r=10, t=50, b=10)


def _fin(fig, h=360):
    fig.update_layout(margin=MARGIN, height=h, legend_title_text="")
    return fig


def churn_donut(eda):
    c = eda["Churn"].value_counts().reindex(["No", "Yes"])
    fig = go.Figure(go.Pie(labels=["Stayed", "Churned"], values=c.values, hole=.62,
                           marker=dict(colors=[C_NO, C_YES]), textinfo="label+percent"))
    fig.update_layout(title="Churn split", showlegend=False)
    return _fin(fig, 340)


def churn_rate_bar(eda, col, title=None, order=None):
    g = eda.groupby(col).agg(rate=("ChurnFlag", "mean"), n=("ChurnFlag", "size")).reset_index()
    g["rate"] = g["rate"] * 100
    if order:
        g[col] = pd.Categorical(g[col], categories=order, ordered=True)
        g = g.sort_values(col)
    fig = px.bar(g, x=col, y="rate", text=g["rate"].map("{:.1f}%".format),
                 hover_data={"n": True, "rate": ":.1f"}, color_discrete_sequence=[C_YES])
    fig.update_traces(textposition="outside")
    fig.update_layout(title=title or f"Churn rate by {col}", yaxis_title="Churn rate (%)",
                      xaxis_title="")
    fig.update_yaxes(range=[0, max(g["rate"].max() * 1.2, 10)])
    return _fin(fig)


def hist_by_churn(eda, col, nbins=30, title=None):
    fig = px.histogram(eda, x=col, color="Churn", nbins=nbins, barmode="overlay",
                       opacity=.65, color_discrete_map=CMAP)
    fig.update_layout(title=title or f"{col} distribution", yaxis_title="Customers")
    return _fin(fig)


def box_by_churn(eda, col, title=None):
    fig = px.box(eda, x="Churn", y=col, color="Churn", color_discrete_map=CMAP, points=False)
    fig.update_layout(title=title or f"{col} by churn", showlegend=False)
    return _fin(fig)


def corr_bar(X: pd.DataFrame, y: pd.Series, top=15):
    c = X.corrwith(y).dropna().sort_values()
    c = pd.concat([c.head(top // 2), c.tail(top - top // 2)])
    colors = [C_NO if v < 0 else C_YES for v in c.values]
    fig = go.Figure(go.Bar(x=c.values, y=[pretty(i) for i in c.index], orientation="h",
                           marker_color=colors))
    fig.update_layout(title="Correlation with churn (+ = more churn)", xaxis_title="Pearson r")
    return _fin(fig, 460)


def heatmap_rate(eda, a, b):
    pv = eda.pivot_table(index=a, columns=b, values="ChurnFlag", aggfunc="mean") * 100
    fig = px.imshow(pv, text_auto=".1f", aspect="auto", color_continuous_scale="OrRd",
                    labels=dict(color="Churn %"))
    fig.update_layout(title=f"Churn rate (%): {a} × {b}")
    return _fin(fig, 420)


def confusion_fig(cm):
    fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", aspect="equal",
                    x=["Pred: Stay", "Pred: Churn"], y=["Actual: Stay", "Actual: Churn"])
    fig.update_layout(title="Confusion matrix", coloraxis_showscale=False)
    return _fin(fig, 340)


def roc_fig(curves: dict):
    fig = go.Figure()
    for name, (y, p) in curves.items():
        fpr, tpr, _ = roc_curve(y, p)
        from sklearn.metrics import roc_auc_score
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                 name=f"{name} (AUC {roc_auc_score(y, p):.3f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                             line=dict(dash="dash", color="gray")))
    fig.update_layout(title="ROC curve", xaxis_title="False positive rate",
                      yaxis_title="True positive rate")
    return _fin(fig, 400)


def pr_fig(y, p):
    pr, rc, _ = precision_recall_curve(y, p)
    fig = go.Figure(go.Scatter(x=rc, y=pr, mode="lines", line=dict(color=C_YES)))
    fig.add_hline(y=float(np.mean(y)), line_dash="dash", line_color="gray",
                  annotation_text="baseline (churn rate)")
    fig.update_layout(title="Precision–Recall curve", xaxis_title="Recall", yaxis_title="Precision")
    return _fin(fig, 400)


def threshold_fig(y, p, current):
    ths = np.linspace(0.02, 0.98, 49)
    from sklearn.metrics import f1_score, precision_score, recall_score
    pr = [precision_score(y, p > t, zero_division=0) for t in ths]
    rc = [recall_score(y, p > t, zero_division=0) for t in ths]
    f1 = [f1_score(y, p > t, zero_division=0) for t in ths]
    fig = go.Figure()
    for n, v in (("Precision", pr), ("Recall", rc), ("F1", f1)):
        fig.add_trace(go.Scatter(x=ths, y=v, mode="lines", name=n))
    fig.add_vline(x=current, line_dash="dot", line_color="gray", annotation_text="current")
    fig.update_layout(title="Metrics vs decision threshold", xaxis_title="Threshold",
                      yaxis_title="Score")
    return _fin(fig, 400)


def importance_fig(imp: pd.Series, top=12):
    s = imp.sort_values().tail(top)
    fig = go.Figure(go.Bar(x=s.values, y=[pretty(i) for i in s.index], orientation="h",
                           marker_color=C_NO))
    fig.update_layout(title=f"Top {len(s)} feature importances", xaxis_title="Importance")
    return _fin(fig, 440)


def depth_fig(df, cols=("Train Acc", "Test Acc"), title="max_depth vs accuracy"):
    fig = go.Figure()
    for c in cols:
        fig.add_trace(go.Scatter(x=df["max_depth"], y=df[c], mode="lines+markers", name=c))
    fig.update_layout(title=title, xaxis_title="max_depth", yaxis_title="Score")
    fig.update_xaxes(dtick=1)
    return _fin(fig, 400)


def gauge(prob, thr=0.5):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=prob * 100, number={"suffix": "%", "valueformat": ".1f"},
        title={"text": "Churn probability"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#333", "thickness": .25},
               "steps": [{"range": [0, 30], "color": "#2FA36B"},
                         {"range": [30, 60], "color": "#E8A33C"},
                         {"range": [60, 100], "color": "#D64545"}],
               "threshold": {"line": {"color": "black", "width": 4}, "value": thr * 100}}))
    fig.update_layout(margin=dict(l=20, r=20, t=60, b=10), height=300)
    return fig


def compare_bars(table: pd.DataFrame, cols):
    long = table.melt(id_vars="Model", value_vars=list(cols), var_name="Metric", value_name="Score")
    fig = px.bar(long, x="Model", y="Score", color="Metric", barmode="group")
    fig.update_layout(title="Model comparison", xaxis_title="")
    fig.update_xaxes(tickangle=-25)
    return _fin(fig, 460)


def bar_simple(labels, values, title, ytitle="Churn probability (%)", highlight=None):
    colors = [C_YES if l == highlight else C_NO for l in labels]
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=colors,
                           text=[f"{v:.1f}%" for v in values], textposition="outside"))
    fig.update_layout(title=title, yaxis_title=ytitle)
    fig.update_yaxes(range=[0, max(max(values) * 1.25, 10)])
    return _fin(fig, 340)


def line_simple(x, y, title, xtitle, current_x=None):
    fig = go.Figure(go.Scatter(x=x, y=y, mode="lines", line=dict(color=C_YES, shape="hv")))
    if current_x is not None:
        fig.add_vline(x=current_x, line_dash="dot", line_color="gray", annotation_text="this customer")
    fig.update_layout(title=title, xaxis_title=xtitle, yaxis_title="Churn probability (%)")
    return _fin(fig, 340)


def impurity_curves():
    p = np.linspace(0.001, 0.999, 300)
    gini = 1 - (p ** 2 + (1 - p) ** 2)
    ent = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=p, y=gini, name="Gini (max 0.5)"))
    fig.add_trace(go.Scatter(x=p, y=ent, name="Entropy (max 1.0)"))
    fig.update_layout(title="Impurity vs share of churners in a node",
                      xaxis_title="p (fraction of class 'Churn')", yaxis_title="Impurity")
    return _fin(fig, 380)
