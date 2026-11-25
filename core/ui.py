"""Small UI helpers: version-safe 'stretch' widgets, hero header, badges."""
import streamlit as st


def _stretch(fn, *args, **kwargs):
    """New Streamlit uses width='stretch', older ones use use_container_width=True."""
    try:
        return fn(*args, width="stretch", **kwargs)
    except Exception:
        return fn(*args, use_container_width=True, **kwargs)


def show_plot(fig, key=None):
    return _stretch(st.plotly_chart, fig, key=key)


def show_df(df, **kwargs):
    return _stretch(st.dataframe, df, **kwargs)


def show_pyplot(fig):
    return _stretch(st.pyplot, fig)


def button(label, **kwargs):
    return _stretch(st.button, label, **kwargs)


def download_button(label, data, file_name, mime, **kwargs):
    return _stretch(st.download_button, label, data=data, file_name=file_name, mime=mime, **kwargs)


def header(title: str, subtitle: str = ""):
    st.markdown(
        f"""<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>""",
        unsafe_allow_html=True,
    )


def risk_badge(prob: float) -> str:
    if prob >= 0.60:
        return '<span class="badge high">HIGH RISK</span>'
    if prob >= 0.30:
        return '<span class="badge mid">MEDIUM RISK</span>'
    return '<span class="badge low">LOW RISK</span>'


def risk_label(prob: float) -> str:
    return "High" if prob >= 0.60 else ("Medium" if prob >= 0.30 else "Low")


CSS = """
<style>
.block-container {padding-top: 1.4rem; max-width: 1450px;}
.hero {background: linear-gradient(135deg, #4C9BE8 0%, #7B6FE0 55%, #E8734C 100%);
       padding: 1.3rem 1.6rem; border-radius: 16px; margin-bottom: 1.1rem;}
.hero h1 {margin: 0; font-size: 1.8rem; color: #fff !important; padding: 0;}
.hero p {margin: .3rem 0 0 0; color: rgba(255,255,255,.92); font-size: 1rem;}
div[data-testid="stMetric"] {background: rgba(128,128,128,.08);
       border: 1px solid rgba(128,128,128,.22); padding: .75rem 1rem; border-radius: 12px;}
.badge {display:inline-block; padding: .25rem .8rem; border-radius: 999px; font-weight: 700;
        font-size: .85rem; letter-spacing: .04em; color: #fff;}
.badge.high {background:#D64545;} .badge.mid {background:#E8A33C;} .badge.low {background:#2FA36B;}
.small-note {font-size:.85rem; opacity:.75;}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)
