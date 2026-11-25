"""Shared model config kept in session_state so every page uses the same model."""
import streamlit as st

from core.model import DEFAULT_CFG

W_KEYS = ["w_criterion", "w_unlimited", "w_depth", "w_split", "w_leaf", "w_cw",
          "w_ccp", "w_thr", "w_test", "w_seed"]


def get_cfg() -> dict:
    return st.session_state.setdefault("cfg", dict(DEFAULT_CFG))


def apply_cfg(cfg: dict):
    """Safe to call from callbacks: replaces config and resets Model Lab widgets."""
    st.session_state["cfg"] = dict(cfg)
    for k in W_KEYS:
        st.session_state.pop(k, None)
