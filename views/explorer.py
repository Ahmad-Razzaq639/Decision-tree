import pandas as pd
import plotly.express as px
import streamlit as st

from core import charts
from core.data import get_eda, load_raw
from core.ui import download_button, header, show_df, show_plot

CAT_COLS = ["gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
            "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
            "PaymentMethod", "TenureGroup"]


def render():
    header("🗂️ Data Explorer", "Filter karo, profile dekho, aur segments ka churn compare karo.")
    eda = get_eda()
    tab1, tab2, tab3, tab4 = st.tabs(["Filter & table", "Column profile", "Segment heatmap", "Data quality"])

    with tab1:
        with st.container(border=True):
            c1, c2, c3, c4 = st.columns(4)
            contract = c1.multiselect("Contract", sorted(eda["Contract"].unique()))
            internet = c2.multiselect("Internet service", sorted(eda["InternetService"].unique()))
            pay = c3.multiselect("Payment method", sorted(eda["PaymentMethod"].unique()))
            churn = c4.multiselect("Churn", ["Yes", "No"])
            s1, s2 = st.columns(2)
            ten = s1.slider("Tenure (months)", 0, 72, (0, 72))
            mon = s2.slider("Monthly charges ($)", 18.0, 120.0, (18.0, 120.0))

        f = eda[eda["tenure"].between(*ten) & eda["MonthlyCharges"].between(*mon)]
        for col, sel in (("Contract", contract), ("InternetService", internet),
                         ("PaymentMethod", pay), ("Churn", churn)):
            if sel:
                f = f[f[col].isin(sel)]

        k1, k2, k3 = st.columns(3)
        k1.metric("Matching customers", f"{len(f):,}", f"{len(f)/len(eda)*100:.1f}% of all")
        k2.metric("Churn rate in selection",
                  f"{f['ChurnFlag'].mean()*100:.1f}%" if len(f) else "–",
                  f"{(f['ChurnFlag'].mean()-eda['ChurnFlag'].mean())*100:+.1f} pp vs overall" if len(f) else None,
                  delta_color="inverse")
        k3.metric("Avg monthly bill", f"${f['MonthlyCharges'].mean():.2f}" if len(f) else "–")

        view = f.drop(columns=["ChurnFlag", "TenureGroup"])
        show_df(view, height=420, hide_index=True)
        download_button("⬇️ Download filtered CSV", view.to_csv(index=False).encode(),
                        "filtered_customers.csv", "text/csv")

    with tab2:
        col = st.selectbox("Choose a column", CAT_COLS + ["tenure", "MonthlyCharges", "TotalCharges"])
        if col in ("tenure", "MonthlyCharges", "TotalCharges"):
            a, b = st.columns(2)
            with a:
                show_plot(charts.hist_by_churn(eda, col, 40))
            with b:
                show_plot(charts.box_by_churn(eda, col))
        else:
            a, b = st.columns(2)
            vc = eda.groupby([col, "Churn"]).size().reset_index(name="Customers")
            with a:
                fig = px.bar(vc, x=col, y="Customers", color="Churn", barmode="group",
                             color_discrete_map=charts.CMAP, title=f"{col}: customer counts")
                show_plot(fig)
            with b:
                show_plot(charts.churn_rate_bar(eda, col))

    with tab3:
        a, b = st.columns(2)
        row = a.selectbox("Rows", CAT_COLS, index=CAT_COLS.index("Contract"))
        col = b.selectbox("Columns", CAT_COLS, index=CAT_COLS.index("InternetService"))
        if row == col:
            st.warning("Rows aur columns alag select karo.")
        else:
            show_plot(charts.heatmap_rate(eda, row, col))

    with tab4:
        raw = load_raw()
        a, b = st.columns(2)
        with a:
            st.markdown("**Missing / blank values (raw file)**")
            blanks = raw.apply(lambda s: s.astype(str).str.strip().eq("").sum() + s.isna().sum())
            show_df(blanks[blanks > 0].rename("count").to_frame()
                    if (blanks > 0).any() else pd.DataFrame({"info": ["No missing values"]}))
            st.caption("TotalCharges mein blank values hain (tenure = 0 wale naye customers). "
                       "Notebook ki tarah inhe median se fill kiya jata hai.")
            st.metric("Duplicate rows", int(raw.duplicated().sum()))
        with b:
            st.markdown("**Numeric summary**")
            show_df(eda[["tenure", "MonthlyCharges", "TotalCharges"]].describe().round(2))
