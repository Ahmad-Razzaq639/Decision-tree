import streamlit as st

from core import charts
from core.data import get_encoded, get_eda
from core.ui import header, show_plot


def render():
    header("📊 Customer Churn Overview",
           "Kaun se customers jaa rahe hain, aur kyun? Dataset ki poori tasveer ek jagah.")
    eda = get_eda()
    enc = get_encoded()

    churned = eda[eda["ChurnFlag"] == 1]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Customers", f"{len(eda):,}")
    c2.metric("Churn rate", f"{eda['ChurnFlag'].mean()*100:.1f}%")
    c3.metric("Avg tenure", f"{eda['tenure'].mean():.1f} mo")
    c4.metric("Avg monthly bill", f"${eda['MonthlyCharges'].mean():.2f}")
    c5.metric("Monthly revenue at risk", f"${churned['MonthlyCharges'].sum():,.0f}")

    st.markdown("#### 🔎 Key insights")
    rate = lambda col, val: eda.loc[eda[col] == val, "ChurnFlag"].mean() * 100
    i1, i2, i3, i4 = st.columns(4)
    i1.info(f"**Month-to-month** contracts churn at **{rate('Contract','Month-to-month'):.1f}%** "
            f"vs **{rate('Contract','Two year'):.1f}%** for two-year.")
    i2.info(f"**Fiber optic** customers churn at **{rate('InternetService','Fiber optic'):.1f}%** "
            f"vs **{rate('InternetService','DSL'):.1f}%** for DSL.")
    i3.info(f"Customers in their **first 12 months** churn at **{rate('TenureGroup','0-12 mo'):.1f}%**.")
    i4.info(f"**Electronic check** users churn at **{rate('PaymentMethod','Electronic check'):.1f}%** "
            f"vs **{rate('PaymentMethod','Credit card (automatic)'):.1f}%** on credit card.")

    a, b, c = st.columns([1, 1.3, 1.3])
    with a:
        show_plot(charts.churn_donut(eda))
    with b:
        show_plot(charts.churn_rate_bar(eda, "Contract",
                  order=["Month-to-month", "One year", "Two year"]))
    with c:
        show_plot(charts.churn_rate_bar(eda, "InternetService", order=["DSL", "Fiber optic", "No"]))

    a, b = st.columns(2)
    with a:
        show_plot(charts.churn_rate_bar(eda, "PaymentMethod"))
    with b:
        show_plot(charts.churn_rate_bar(eda, "TenureGroup",
                  order=["0-12 mo", "13-24 mo", "25-48 mo", "49-72 mo"],
                  title="Churn rate by tenure group"))

    a, b = st.columns(2)
    with a:
        show_plot(charts.hist_by_churn(eda, "tenure", 36, "Tenure (months) distribution"))
    with b:
        show_plot(charts.hist_by_churn(eda, "MonthlyCharges", 40, "Monthly charges distribution"))

    a, b = st.columns([1, 1])
    with a:
        show_plot(charts.box_by_churn(eda, "TotalCharges", "Total charges by churn"))
    with b:
        X = enc.drop(columns=["Churn"])
        show_plot(charts.corr_bar(X, enc["Churn"]))

    with st.expander("More segments: senior citizens, partner, paperless billing, tech support"):
        cols = st.columns(4)
        for col, name in zip(cols, ["SeniorCitizen", "Partner", "PaperlessBilling", "TechSupport"]):
            with col:
                show_plot(charts.churn_rate_bar(eda, name, name))
