import re as _re

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from kda_backend import ALL_METHODS, run_kda

PAGE_STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;900&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

/* ===== GBK Brand Colors =====
Dark Blue  #3B4954
Red        #F76362
Light Blue #C7D8E4
Blue       #789FC0
Light Red  #F9BDBC
White      #FFFFFF
============================= */

.stApp { background-color: #3B4954 !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1rem 2rem 2rem !important; max-width: 100% !important; }

.gbk-hero { background: #2F3F4B; padding: 2rem 2rem 1.75rem; border-radius: 10px; margin-bottom: 1.5rem; }
.gbk-eyebrow { font-size: 11px; color: #F76362; letter-spacing: 3px; text-transform: uppercase; font-weight: 600; margin-bottom: 0.5rem; }
.gbk-hero h1 { font-size: 36px; font-weight: 900; color: #FFFFFF; line-height: 1.05; text-transform: uppercase; letter-spacing: -1px; margin: 0 0 0.5rem; }
.gbk-hero p { font-size: 13px; color: rgba(255,255,255,0.82); line-height: 1.7; margin: 0; }

.gbk-label { font-size: 10px; color: rgba(255,255,255,0.70); text-transform: uppercase; letter-spacing: 2.5px; font-weight: 700; margin-bottom: 0.4rem; }
.gbk-panel { background: #334651; border-radius: 10px; border: 1px solid rgba(199,216,228,0.18); padding: 1.25rem 1.5rem; margin-bottom: 1rem; }
.gbk-panel-title { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 2.5px; color: rgba(255,255,255,0.62); margin-bottom: 0.75rem; }
.gbk-note { font-size: 13px; color: rgba(255,255,255,0.84); line-height: 1.7; }
.gbk-stat { font-size: 30px; font-weight: 800; color: #FFFFFF; line-height: 1; }
.gbk-card { background: #334651; border-radius: 10px; border: 1px solid rgba(199,216,228,0.18); padding: 1.25rem 1.5rem; }
.gbk-card-kicker { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 2.5px; color: rgba(255,255,255,0.58); margin-bottom: 0.6rem; }
.gbk-card-text { font-size: 18px; font-weight: 800; color: #FFFFFF; line-height: 1.2; }

.gbk-bar-wrap { margin-bottom: 10px; }
.gbk-bar-label { font-size: 12px; color: rgba(255,255,255,0.90); margin-bottom: 3px; }
.gbk-bar-row { display: flex; align-items: center; gap: 10px; }
.gbk-bar-track { flex: 1; background: rgba(255,255,255,0.12); border-radius: 4px; height: 8px; overflow: hidden; }
.gbk-bar-fill { height: 100%; border-radius: 4px; }
.gbk-bar-val { font-size: 11px; color: rgba(255,255,255,0.68); width: 48px; text-align: right; }
.gbk-disclaimer { font-size: 11px; color: rgba(255,255,255,0.52); margin-top: 0.75rem; font-style: italic; }

.gbk-insight { background: rgba(255,255,255,0.05); border-radius: 6px; padding: 0.75rem 1rem; border-left: 3px solid rgba(199,216,228,0.24); font-size: 13px; color: rgba(255,255,255,0.84); line-height: 1.6; margin-bottom: 8px; }
.gbk-insight b { color: #FFFFFF; }
.gbk-insight-red { border-left-color: #F76362; }
.gbk-insight-blue { border-left-color: #789FC0; }

.gbk-step-item { display: flex; gap: 10px; align-items: flex-start; font-size: 13px; color: rgba(255,255,255,0.84); line-height: 1.6; margin-bottom: 10px; }
.gbk-step-num { background: #F76362; color: #FFFFFF; font-size: 10px; font-weight: 700; min-width: 20px; height: 20px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-top: 2px; flex-shrink: 0; }
.gbk-step-item b { color: #FFFFFF; }

.gbk-summary-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
.gbk-summary-key { font-size: 10px; color: rgba(255,255,255,0.58); text-transform: uppercase; letter-spacing: 2px; margin-bottom: 4px; }
.gbk-summary-val { font-size: 13px; color: #FFFFFF; font-weight: 600; }

.gbk-tag { display: inline-block; background: rgba(120,159,192,0.20); color: #C7D8E4; font-size: 12px; padding: 2px 9px; border-radius: 4px; margin: 2px; }
.gbk-method-box { background: rgba(255,255,255,0.05); border: 1px solid rgba(199,216,228,0.18); border-radius: 8px; padding: 10px 14px; margin-top: 8px; }
.gbk-method-title { font-size: 13px; color: #FFFFFF; font-weight: 700; margin-bottom: 4px; }
.gbk-method-desc { font-size: 12px; color: rgba(255,255,255,0.80); line-height: 1.55; }
.gbk-shap-badge { display: inline-block; background: rgba(247,99,98,0.16); border: 1px solid rgba(247,99,98,0.42); color: #F76362; font-size: 10px; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase; padding: 1px 7px; border-radius: 4px; margin-left: 8px; vertical-align: middle; }
.gbk-input-warning { font-size: 11px; color: #F76362; margin-top: 6px; font-weight: 600; }
.gbk-warning-card { background: rgba(247,99,98,0.10); border: 1px solid rgba(247,99,98,0.35); border-left: 4px solid #F76362; border-radius: 8px; padding: 0.8rem 1rem; margin: 0.6rem 0; color: rgba(255,255,255,0.86); font-size: 13px; line-height: 1.55; }
.gbk-warning-card b { color: #FFFFFF; }
.gbk-subtle-rule { border-top: 1px solid rgba(199,216,228,0.16); margin: 0.75rem 0; }

/* Streamlit widget labels on the dark GBK surface */
label,
label p,
label span,
div[data-testid="stWidgetLabel"],
div[data-testid="stWidgetLabel"] p,
div[data-testid="stWidgetLabel"] span,
div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] li {
  color: rgba(255,255,255,0.86) !important;
}
div[data-testid="stWidgetLabel"] {
  margin-bottom: 0.35rem !important;
}
div[data-testid="InputInstructions"],
div[data-testid="stTooltipHoverTarget"],
div[data-testid="stHelp"] {
  color: rgba(199,216,228,0.78) !important;
}

div[data-testid="stButton"] > button,
div[data-testid="stFormSubmitButton"] > button {
  background: #F76362 !important;
  color: #FFFFFF !important;
  border: none !important;
  border-radius: 8px !important;
  font-size: 12px !important;
  font-weight: 700 !important;
  letter-spacing: 1.2px !important;
  text-transform: uppercase !important;
  padding: 0.6rem 1rem !important;
}
div[data-testid="stButton"] > button:hover,
div[data-testid="stFormSubmitButton"] > button:hover {
  background: #e55a59 !important;
}

div[data-baseweb="select"] > div,
div[data-testid="stSelectbox"] > div > div {
  background: #334651 !important;
  border: 1px solid rgba(199,216,228,0.24) !important;
  border-radius: 8px !important;
  color: #FFFFFF !important;
}
div[data-baseweb="select"] span,
div[data-baseweb="select"] input,
div[data-testid="stSelectbox"] span,
div[data-testid="stSelectbox"] input,
div[data-testid="stMultiSelect"] span,
div[data-testid="stMultiSelect"] input {
  color: #FFFFFF !important;
  -webkit-text-fill-color: #FFFFFF !important;
}
div[data-baseweb="select"] input::placeholder,
div[data-testid="stSelectbox"] input::placeholder,
div[data-testid="stMultiSelect"] input::placeholder {
  color: rgba(255,255,255,0.64) !important;
  -webkit-text-fill-color: rgba(255,255,255,0.64) !important;
  opacity: 1 !important;
}
div[data-baseweb="popover"],
div[data-baseweb="popover"] *,
div[data-baseweb="menu"],
div[data-baseweb="menu"] *,
div[role="listbox"],
div[role="listbox"] *,
ul[role="listbox"],
ul[role="listbox"] *,
li[role="option"],
li[role="option"] * {
  color: #1F2D36 !important;
  -webkit-text-fill-color: #1F2D36 !important;
}
div[data-baseweb="popover"] div[aria-selected="true"],
div[data-baseweb="popover"] div[aria-selected="true"] *,
div[role="listbox"] div[aria-selected="true"],
div[role="listbox"] div[aria-selected="true"] * {
  background: #E9EEF2 !important;
  color: #1F2D36 !important;
  -webkit-text-fill-color: #1F2D36 !important;
}
div[data-testid="stMultiSelect"] > div {
  background: #334651 !important;
  border: 1px solid rgba(199,216,228,0.24) !important;
  border-radius: 8px !important;
}
div[data-testid="stMultiSelect"] div[data-baseweb="select"] > div,
div[data-testid="stMultiSelect"] div[data-baseweb="select"] div,
div[data-testid="stMultiSelect"] div[data-baseweb="select"] span,
div[data-testid="stMultiSelect"] div[data-baseweb="select"] input {
  color: #FFFFFF !important;
  -webkit-text-fill-color: #FFFFFF !important;
}
div[data-testid="stMultiSelect"] div[data-baseweb="select"] input::placeholder,
div[data-testid="stMultiSelect"] div[data-baseweb="select"] [aria-disabled="true"],
div[data-testid="stMultiSelect"] div[data-baseweb="select"] [aria-disabled="true"] * {
  color: rgba(255,255,255,0.74) !important;
  -webkit-text-fill-color: rgba(255,255,255,0.74) !important;
  opacity: 1 !important;
}

div[data-testid="stNumberInput"] label,
div[data-testid="stNumberInput"] label p,
div[data-testid="stNumberInput"] label span {
  color: rgba(255,255,255,0.86) !important;
}
div[data-testid="stNumberInput"] input {
  color: #1F2D36 !important;
  -webkit-text-fill-color: #1F2D36 !important;
}
div[data-testid="stNumberInput"] button {
  color: #1F2D36 !important;
}

div[data-testid="stFileUploader"] section {
  background: #334651 !important;
  border: 1px dashed rgba(199,216,228,0.28) !important;
  border-radius: 10px !important;
}
div[data-testid="stFileUploader"] small,
div[data-testid="stFileUploader"] span,
div[data-testid="stFileUploader"] label,
div[data-testid="stFileUploader"] p {
  color: rgba(255,255,255,0.80) !important;
}

details {
  background: #334651 !important;
  border: 1px solid rgba(199,216,228,0.18) !important;
  border-radius: 10px !important;
  margin-bottom: 1rem !important;
}
details[open] {
  background: #334651 !important;
}
details summary {
  font-size: 12px !important;
  background: #334651 !important;
  color: rgba(255,255,255,0.88) !important;
  padding: 0.75rem 1rem !important;
  border-radius: 10px !important;
}
details summary *,
details summary p,
details summary span,
details summary svg {
  color: rgba(255,255,255,0.88) !important;
  fill: rgba(255,255,255,0.88) !important;
}
details > div {
  background: #334651 !important;
}

div[data-testid="stDataFrame"] { background: #334651 !important; border-radius: 8px !important; }

.stMarkdown p, .stMarkdown li { color: rgba(255,255,255,0.84) !important; }
.stMarkdown strong, .stMarkdown b { color: #FFFFFF !important; }

/* Checkbox text + tick */
div[data-testid="stCheckbox"] { margin-bottom: 0.25rem; }
div[data-testid="stCheckbox"] label,
div[data-testid="stCheckbox"] label p,
div[data-testid="stCheckbox"] label span,
div[data-testid="stCheckbox"] [data-testid="stMarkdownContainer"] p {
  color: #C7D8E4 !important;
  opacity: 1 !important;
}
div[data-testid="stCheckbox"] input:checked {
  accent-color: #F76362 !important;
}
</style>
"""


def configure_page():
    st.set_page_config(page_title="GBK Marketing Insights Suite", layout="wide")
    for k, v in {
        "uploaded_df_raw": None,
        "uploaded_df_num": None,
        "uploaded_meta": None,
        "uploaded_filename": None,
        "uploaded_sheet_name": None,
        "uploaded_sheet_names": None,
        "uploaded_file_signature": None,
        "analysis_result": None,
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v
    st.markdown(PAGE_STYLE, unsafe_allow_html=True)

NAME_MAP = {
    "C5_FINALr6": "Overall Satisfaction",
    "C6": "Product Quality",
    "C7": "Ease of Use",
    "C8": "Price / Value",
    "C9": "Customer Support",
    "C10": "Purchase Experience",
    "C11": "Brand Trust",
    "C12": "Retailer",
}
BAR_COLORS = ["#F76362", "#C7D8E4", "#789FC0", "#F9BDBC", "#5E7486"]
DEFAULT_METHODS = ("correlation", "regression")
DEFAULT_BOOTSTRAP_METHODS = ("correlation", "regression", "shapley_lmg", "johnson")
HEAVY_BOOTSTRAP_METHODS = ("random_forest", "xgboost", "shap")
DEFAULT_BOOTSTRAP_RESAMPLES = 40
METHOD_COLORS = {
    "correlation": "#F76362",
    "regression": "#C7D8E4",
    "drop_one": "#3B4954",
    "shapley_lmg": "#D76BA6",
    "johnson": "#2FA872",
    "random_forest": "#8B6BD6",
    "xgboost": "#F0B35A",
    "shap": "#2CA6A4",
}

METHOD_LABELS = {
    "correlation": "Correlation",
    "regression": "Regression",
    "drop_one": "Drop-one",
    "shapley_lmg": "Shapley / LMG",
    "johnson": "Johnson Relative Weights",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "shap": "SHAP",
}

METHOD_INFO = {
    "shap": {
        "title": "SHAP — Shapley Values",
        "recommended": True,
        "desc": (
            "Ranks drivers by average absolute feature contribution after fitting an XGBoost model. "
            "Useful for non-linear patterns and client-friendly variable importance summaries."
        ),
        "note": "Mean |SHAP value| — average absolute impact on outcome.",
    },
    "random_forest": {
        "title": "Random Forest Importance",
        "recommended": False,
        "desc": (
            "Good for non-linear patterns. Trains an ensemble of decision trees and ranks variables by how "
            "much they reduce prediction error. Reliable and fast, but scores can be biased toward "
            "high-cardinality variables. Good as a comparison check against SHAP."
        ),
        "note": "Permutation importance from Random Forest.",
    },
    "correlation": {
        "title": "Correlation (Pearson |r|)",
        "recommended": False,
        "desc": (
            "Simple and transparent. Ranks variables by absolute linear correlation with the outcome. "
            "Easy to explain to non-technical clients, but misses non-linear effects and ignores "
            "interactions between variables."
        ),
        "note": "Pearson |r| — absolute linear correlation with outcome.",
    },
    "regression": {
        "title": "Standardized Regression Coefficients",
        "recommended": False,
        "desc": (
            "Good for linear relationships. Matches the GBK/R style standardized OLS beta calculation "
            "on numeric outcome coding, so binary consideration outcomes remain comparable with the "
            "client reference workbook. Interpretable, but unstable when predictors are highly correlated."
        ),
        "note": "Standardized coefficient β — signed linear importance.",
    },
    "drop_one": {
        "title": "Drop-one Usefulness",
        "recommended": False,
        "desc": "Compares the full model against reduced models that remove one driver at a time.",
        "note": "Full model score minus reduced model score.",
    },
    "shapley_lmg": {
        "title": "Shapley / LMG",
        "recommended": False,
        "desc": "Allocates linear model explained variance across drivers using numeric outcome coding.",
        "note": "LMG share of model R-squared.",
    },
    "johnson": {
        "title": "Johnson Relative Weights",
        "recommended": False,
        "desc": "Decomposes linear model explanatory power under correlated predictors using numeric outcome coding.",
        "note": "Johnson relative weight.",
    },
    "xgboost": {
        "title": "XGBoost Importance",
        "recommended": False,
        "desc": "Fits a gradient boosted tree model and ranks drivers by gain-based feature importance.",
        "note": "XGBoost gain importance.",
    },
}


def _auto_label(col):
    s = str(col).replace("_", " ")
    s = _re.sub(r"([a-zA-Z])(\d)", r"\1 \2", s)
    s = _re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    return _re.sub(r"\s+", " ", s).strip().title()

def display_name(col):
    return NAME_MAP.get(col, _auto_label(col))

def is_excluded_column(col):
    return any(k in str(col).lower() for k in [
        "uuid", "record", "date", "start_date", "psid", "pid",
        "marker", "status", "qualityscore", "linercheck", "loi"
    ])

def is_lookup_column(col):
    name = str(col).lower()
    return (
        name.endswith("_code")
        or name in {"brand_code", "top_brand_name", "ownership_brand_name"}
        or name.startswith("top_brand")
        or name.startswith("ownership_brand")
    )

def is_outcome_column(col):
    name = str(col).lower()
    return "consideration" in name or "satisfaction" in name

def is_segment_column(col):
    name = str(col).lower()
    segment_tokens = [
        "gender",
        "age",
        "region",
        "ecosystem",
        "channel",
        "frequency",
        "segment",
        "country",
        "market",
    ]
    return any(token in name for token in segment_tokens)

def is_driver_column(col):
    if is_excluded_column(col) or is_lookup_column(col) or is_outcome_column(col) or is_segment_column(col):
        return False
    return str(col).lower() != "brand"

def is_subgroup_candidate(df, col):
    if is_excluded_column(col) or is_lookup_column(col) or is_outcome_column(col):
        return False
    unique = df[col].nunique(dropna=True)
    if unique < 2 or unique > 20:
        return False
    if str(col).lower() == "brand":
        return True
    if is_segment_column(col):
        return True
    return df[col].dtype == object and unique <= 12

def prepare_model_data(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    excluded_cols = [c for c in df.columns if is_excluded_column(c)]
    df_model = df.drop(columns=excluded_cols, errors="ignore")
    df_num = df_model.select_dtypes(include=["number"]).copy()
    outcome_candidates = [c for c in df_num.columns if is_outcome_column(c)]
    driver_candidates = [c for c in df_num.columns if is_driver_column(c)]
    subgroup_candidates = [c for c in df_model.columns if is_subgroup_candidate(df_model, c)]

    missing_ratio = df_num.isna().mean()
    high_missing = missing_ratio[missing_ratio > 0.4].index.tolist()
    drop_missing_cols = []
    for col in high_missing:
        if df_num[col].notna().sum() <= 30:
            drop_missing_cols.append(col)
    df_num = df_num.drop(columns=drop_missing_cols, errors="ignore")
    constant_cols = df_num.columns[df_num.nunique(dropna=True) <= 1].tolist()
    df_num = df_num.drop(columns=constant_cols, errors="ignore")
    outcome_candidates = [c for c in outcome_candidates if c in df_num.columns]
    driver_candidates = [c for c in driver_candidates if c in df_num.columns]
    for col in df_num.columns:
        df_num[col] = df_num[col].fillna(df_num[col].median())
    return df, df_num, {
        "excluded_cols": excluded_cols,
        "drop_missing_cols": drop_missing_cols,
        "subgroup_candidates": subgroup_candidates,
        "outcome_candidates": outcome_candidates,
        "driver_candidates": driver_candidates,
        "constant_cols": constant_cols,
    }

def pill_tags(items):
    if not items:
        return '<span style="color:rgba(255,255,255,0.3);font-size:13px;">None</span>'
    return "".join(f'<span class="gbk-tag">{x}</span>' for x in items)

def render_method_info_box(method_key):
    info = METHOD_INFO.get(method_key)
    if not info:
        return
    badge = '<span class="gbk-shap-badge">Recommended</span>' if info["recommended"] else ""
    st.markdown(
        f'<div class="gbk-method-box">'
        f'<div class="gbk-method-title">{info["title"]}{badge}</div>'
        f'<div class="gbk-method-desc">{info["desc"]}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

def render_bar_chart(driver_scores, title="Driver Importance", method_key=""):
    max_val = driver_scores.iloc[0] if len(driver_scores) else 1
    bars_html = ""
    for i, (col, val) in enumerate(driver_scores.items()):
        color = BAR_COLORS[i] if i < len(BAR_COLORS) else BAR_COLORS[-1]
        pct = round(val / max_val * 100) if max_val else 0
        bars_html += (
            f'<div class="gbk-bar-wrap">'
            f'<div class="gbk-bar-label">{display_name(col)}</div>'
            f'<div class="gbk-bar-row">'
            f'<div class="gbk-bar-track"><div class="gbk-bar-fill" style="width:{pct}%;background:{color};"></div></div>'
            f'<div class="gbk-bar-val">{val:.3f}</div>'
            f'</div></div>'
        )
    note = METHOD_INFO.get(method_key, {}).get("note", "Longer bar = stronger importance.")
    st.markdown(
        f'<div class="gbk-panel"><div class="gbk-panel-title">{title}</div>{bars_html}'
        f'<div class="gbk-disclaimer">{note} Directional, not causal.</div></div>',
        unsafe_allow_html=True
    )

def build_driver_interval_chart(importance_table, methods):
    method_cols = [method for method in methods if method in importance_table.columns]
    if not method_cols:
        method_cols = ["mean_method_index"] if "mean_method_index" in importance_table.columns else []
    plot_df = importance_table.copy()
    if "mean_method_index" in plot_df.columns:
        plot_df = plot_df.sort_values("mean_method_index", ascending=True)
    elif method_cols:
        plot_df = plot_df.sort_values(method_cols[0], ascending=True)

    fig_height = max(4.5, 0.5 * len(plot_df) + 1.5)
    fig, ax = plt.subplots(figsize=(10.5, fig_height))
    y_pos = np.arange(len(plot_df))
    offsets = np.linspace(-0.18, 0.18, max(len(method_cols), 1))

    for idx, method in enumerate(method_cols):
        scores = plot_df[method].to_numpy(dtype=float)
        y = y_pos + offsets[idx]
        color = METHOD_COLORS.get(method, BAR_COLORS[idx % len(BAR_COLORS)])
        lower_col = f"{method}_ci_lower"
        upper_col = f"{method}_ci_upper"
        if lower_col in plot_df.columns and upper_col in plot_df.columns:
            lower = plot_df[lower_col].to_numpy(dtype=float)
            upper = plot_df[upper_col].to_numpy(dtype=float)
            left_err = np.where(np.isfinite(lower), scores - lower, 0)
            right_err = np.where(np.isfinite(upper), upper - scores, 0)
            ax.errorbar(
                scores,
                y,
                xerr=np.vstack([np.maximum(left_err, 0), np.maximum(right_err, 0)]),
                fmt="o",
                color=color,
                ecolor=color,
                elinewidth=1.4,
                capsize=3,
                markersize=5,
                label=METHOD_LABELS.get(method, method),
                alpha=0.95,
            )
        else:
            ax.scatter(scores, y, color=color, s=28, label=METHOD_LABELS.get(method, method), alpha=0.95)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([display_name(driver) for driver in plot_df["driver"]])
    ax.set_xlabel("Importance score")
    ax.set_ylabel("Driver")
    ax.set_title("Driver Importance with Bootstrap Confidence Intervals")
    ax.grid(axis="x", color="#D8E2EA", alpha=0.45)
    ax.set_facecolor("#FFFFFF")
    fig.patch.set_facecolor("#FFFFFF")
    if len(method_cols) > 1:
        ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    return fig

def _normalize_like_main_scores(values, main_scores):
    finite = main_scores.replace([np.inf, -np.inf], np.nan).dropna()
    out = pd.Series(np.nan, index=values.index, dtype=float)
    if finite.empty:
        return out
    mean_v = finite.mean()
    if np.isclose(float(mean_v), 0.0):
        abs_mean = finite.abs().mean()
        if np.isclose(float(abs_mean), 0.0):
            return out
        out.loc[values.index] = values / abs_mean * 100.0
    else:
        out.loc[values.index] = values / mean_v * 100.0
    return out

def build_interactive_chart_data(importance_table, methods):
    rows = []
    table = importance_table.copy()
    sort_col = "mean_method_index" if "mean_method_index" in table.columns else methods[0]
    table = table.sort_values(sort_col, ascending=False).reset_index(drop=True)
    driver_order = table["driver"].tolist()

    for method in methods:
        if method not in table.columns:
            continue
        score_col = f"{method}_index"
        scores = table[score_col] if score_col in table.columns else table[method]
        lower = pd.Series(np.nan, index=table.index, dtype=float)
        upper = pd.Series(np.nan, index=table.index, dtype=float)
        if f"{method}_ci_lower" in table.columns and f"{method}_ci_upper" in table.columns:
            main_scores = table[method]
            lower = _normalize_like_main_scores(table[f"{method}_ci_lower"], main_scores)
            upper = _normalize_like_main_scores(table[f"{method}_ci_upper"], main_scores)
        for idx, row in table.iterrows():
            rows.append(
                {
                    "driver": row["driver"],
                    "driver_label": display_name(row["driver"]),
                    "driver_order": driver_order.index(row["driver"]),
                    "method": METHOD_LABELS.get(method, method),
                    "method_key": method,
                    "score": float(scores.iloc[idx]) if pd.notna(scores.iloc[idx]) else np.nan,
                    "raw_score": float(row[method]) if pd.notna(row[method]) else np.nan,
                    "ci_lower": float(lower.iloc[idx]) if pd.notna(lower.iloc[idx]) else np.nan,
                    "ci_upper": float(upper.iloc[idx]) if pd.notna(upper.iloc[idx]) else np.nan,
                }
            )
    return pd.DataFrame(rows)

def _driver_axis_sort(chart_df):
    return (
        chart_df[["driver_label", "driver_order"]]
        .drop_duplicates()
        .sort_values("driver_order", ascending=True)["driver_label"]
        .tolist()
    )

def build_interactive_driver_chart(importance_table, methods, x_domain_override=None):
    chart_df = build_interactive_chart_data(importance_table, methods)
    if chart_df.empty:
        return None

    domain = [METHOD_LABELS.get(method, method) for method in methods]
    chart_colors = [METHOD_COLORS.get(method, "#789FC0") for method in methods]
    y_sort = _driver_axis_sort(chart_df)
    finite_values = pd.concat(
        [chart_df["score"], chart_df["ci_lower"].dropna(), chart_df["ci_upper"].dropna()],
        ignore_index=True,
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if finite_values.empty:
        x_domain = [0, 100]
    else:
        min_x = min(0.0, float(finite_values.min()))
        max_x = max(100.0, float(finite_values.max()))
        pad = max((max_x - min_x) * 0.08, 5.0)
        x_domain = [min_x - pad, max_x + pad]
    if x_domain_override:
        x_domain = [float(x_domain_override[0]), float(x_domain_override[1])]
    nearest = alt.selection_point(name="method_filter", fields=["method"], bind="legend")
    base = alt.Chart(chart_df).encode(
        y=alt.Y("driver_label:N", sort=y_sort, title="Driver"),
        color=alt.Color("method:N", scale=alt.Scale(domain=domain, range=chart_colors), title="Method"),
        opacity=alt.condition(nearest, alt.value(1), alt.value(0.04)),
        tooltip=[
            alt.Tooltip("driver_label:N", title="Driver"),
            alt.Tooltip("method:N", title="Method"),
            alt.Tooltip("score:Q", title="Index score", format=".1f"),
            alt.Tooltip("raw_score:Q", title="Raw score", format=".4f"),
            alt.Tooltip("ci_lower:Q", title="CI lower index", format=".1f"),
            alt.Tooltip("ci_upper:Q", title="CI upper index", format=".1f"),
        ],
    )
    ci_df = chart_df.dropna(subset=["ci_lower", "ci_upper"])
    ci = alt.Chart(ci_df).mark_rule(strokeWidth=2).encode(
        y=alt.Y("driver_label:N", sort=y_sort, title="Driver"),
        x=alt.X("ci_lower:Q", title="Indexed score (average = 100)", scale=alt.Scale(domain=x_domain, zero=False)),
        x2="ci_upper:Q",
        color=alt.Color("method:N", scale=alt.Scale(domain=domain, range=chart_colors), title="Method"),
        opacity=alt.condition(nearest, alt.value(0.75), alt.value(0.04)),
        tooltip=[
            alt.Tooltip("driver_label:N", title="Driver"),
            alt.Tooltip("method:N", title="Method"),
            alt.Tooltip("ci_lower:Q", title="CI lower index", format=".1f"),
            alt.Tooltip("ci_upper:Q", title="CI upper index", format=".1f"),
        ],
    )
    points = base.mark_circle(size=90).encode(
        x=alt.X("score:Q", title="Indexed score (average = 100)", scale=alt.Scale(domain=x_domain, zero=False)),
    ).add_params(nearest)
    return (ci + points).properties(height=max(360, 34 * len(y_sort)), width="container")

def render_interval_chart(kda_result, methods, title="Driver importance", chart_x_domain=None):
    st.markdown(f'<div class="gbk-panel-title">{title}</div>', unsafe_allow_html=True)
    chart = build_interactive_driver_chart(kda_result.importance_table, methods, chart_x_domain)
    if chart is not None:
        st.altair_chart(chart, width="stretch")
    ci_methods = [
        method
        for method in methods
        if f"{method}_ci_lower" in kda_result.importance_table.columns
        and f"{method}_ci_upper" in kda_result.importance_table.columns
    ]
    if ci_methods:
        labels = ", ".join(METHOD_LABELS.get(method, method) for method in ci_methods)
        st.markdown(
            f'<div class="gbk-disclaimer">Dots are indexed method scores. Horizontal lines are bootstrap confidence intervals for {labels}. The raw scores remain in the export table.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="gbk-disclaimer">Dots are indexed method scores. Enable bootstrap before running analysis to add confidence intervals.</div>',
            unsafe_allow_html=True,
        )

def chart_range_control(kda_result, methods, key_prefix):
    chart_df = build_interactive_chart_data(kda_result.importance_table, methods)
    if chart_df.empty:
        return None
    finite_values = pd.concat(
        [chart_df["score"], chart_df["ci_lower"].dropna(), chart_df["ci_upper"].dropna()],
        ignore_index=True,
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if finite_values.empty:
        return None
    min_x = min(0.0, float(finite_values.min()))
    max_x = max(100.0, float(finite_values.max()))
    pad = max((max_x - min_x) * 0.12, 10.0)
    full_domain = (float(np.floor(min_x - pad)), float(np.ceil(max_x + pad)))
    with st.expander("Chart controls"):
        auto_range = st.checkbox("Auto x-axis range", value=True, key=f"{key_prefix}_auto_range")
        if auto_range:
            return None
        return st.slider(
            "Visible index range",
            min_value=full_domain[0],
            max_value=full_domain[1],
            value=full_domain,
            step=1.0,
            key=f"{key_prefix}_x_range",
        )

def render_insights(target, driver_scores):
    names = [display_name(x) for x in driver_scores.index]
    if not names:
        return
    t = display_name(target)
    n2 = names[1] if len(names) > 1 else names[0]
    n3 = names[2] if len(names) > 2 else n2
    n4 = names[3] if len(names) > 3 else n3
    st.markdown(
        f'<div class="gbk-panel"><div class="gbk-panel-title">What the data suggests</div>'
        f'<div class="gbk-insight gbk-insight-red"><b>Primary signal</b><br>'
        f'<b>{names[0]}</b> appears to be the strongest variable associated with <b>{t}</b>.</div>'
        f'<div class="gbk-insight gbk-insight-blue"><b>Secondary signal</b><br>'
        f'<b>{n2}</b> also shows a meaningful association with this outcome.</div>'
        f'<div class="gbk-insight"><b>Broader context</b><br>'
        f'<b>{n3}</b> and <b>{n4}</b> are also worth reviewing alongside business context and prior knowledge.</div>'
        f'</div>',
        unsafe_allow_html=True
    )

def render_next_steps(target, driver_scores):
    names = [display_name(x) for x in driver_scores.index]
    if not names:
        return
    t = display_name(target)
    n2 = names[1] if len(names) > 1 else names[0]
    n3 = names[2] if len(names) > 2 else n2
    n4 = names[3] if len(names) > 3 else n3
    steps = [
        f"<b>Start with {names[0]}.</b> It shows the strongest association with {t}.",
        f"<b>Review {n2} next.</b> It appears to be another meaningful driver candidate.",
        f"<b>Keep {n3} and {n4} in the discussion.</b> They may matter depending on audience and context.",
        "<b>Validate with business judgment.</b> Use these results as decision support, not a standalone answer.",
    ]
    items = "".join(
        f'<div class="gbk-step-item"><div class="gbk-step-num">{i+1}</div><div>{s}</div></div>'
        for i, s in enumerate(steps)
    )
    st.markdown(
        f'<div class="gbk-panel"><div class="gbk-panel-title">Suggested next steps</div>{items}</div>',
        unsafe_allow_html=True
    )

def render_detail_table(ranked):
    rows = "".join(
        f"<tr>"
        f"<td style='color:rgba(255,255,255,0.25);padding:6px 8px;'>{i+1}</td>"
        f"<td style='color:rgba(255,255,255,0.7);padding:6px 8px;'>{display_name(col)}</td>"
        f"<td style='color:rgba(255,255,255,0.35);padding:6px 8px;'>{val:.3f}</td>"
        f"</tr>"
        for i, (col, val) in enumerate(ranked.items())
    )
    st.markdown(
        f'<div class="gbk-panel"><div class="gbk-panel-title">Full Driver Ranking</div>'
        f'<table style="width:100%;border-collapse:collapse;font-size:13px;"><thead><tr>'
        f'<th style="text-align:left;font-size:10px;color:rgba(255,255,255,0.3);letter-spacing:2px;text-transform:uppercase;padding:0 8px 8px;border-bottom:1px solid rgba(255,255,255,0.08);">#</th>'
        f'<th style="text-align:left;font-size:10px;color:rgba(255,255,255,0.3);letter-spacing:2px;text-transform:uppercase;padding:0 8px 8px;border-bottom:1px solid rgba(255,255,255,0.08);">Driver</th>'
        f'<th style="text-align:left;font-size:10px;color:rgba(255,255,255,0.3);letter-spacing:2px;text-transform:uppercase;padding:0 8px 8px;border-bottom:1px solid rgba(255,255,255,0.08);">Score</th>'
        f'</tr></thead><tbody>{rows}</tbody></table></div>',
        unsafe_allow_html=True
    )

def _ranking_to_series(ranking_table):
    return pd.Series(
        ranking_table["mean_method_index"].to_numpy(),
        index=ranking_table["driver"],
    ).sort_values(ascending=False)

def _importance_to_series(importance_table):
    table = importance_table.copy()
    if "mean_method_index" in table.columns:
        score_col = "mean_method_index"
    else:
        method_cols = [
            col for col in table.columns
            if col not in {"driver", "average_rank", "median_rank", "top3_appearances"}
            and not col.endswith(("_index", "_rank", "_warning", "_ci_lower", "_ci_upper"))
            and pd.api.types.is_numeric_dtype(table[col])
        ]
        if not method_cols:
            return pd.Series(dtype=float)
        score_col = method_cols[0]
    return pd.Series(table[score_col].to_numpy(), index=table["driver"]).sort_values(ascending=False)

def _importance_export_table(kda_result):
    table = kda_result.importance_table.copy()
    if "mean_method_index" in table.columns:
        table = table.sort_values(["mean_method_index", "driver"], ascending=[False, True], na_position="last")
    else:
        table = table.sort_values("driver")
    return table.reset_index(drop=True)

def _subgroup_summary_table(kda_result):
    if kda_result.subgroup_summary is None:
        return pd.DataFrame(columns=["subgroup_level", "rows_used", "status", "reason"])
    cols = ["subgroup_level", "rows_used", "status", "reason"]
    return kda_result.subgroup_summary[cols].copy()

def _combined_subgroup_export_table(kda_result, subgroup_var):
    rows = []
    for level, subgroup_result in (kda_result.subgroup_results or {}).items():
        table = _importance_export_table(subgroup_result).copy()
        table.insert(0, "subgroup_level", level)
        table.insert(0, "subgroup_variable", subgroup_var)
        rows.append(table)
    if not rows:
        return pd.DataFrame(columns=["subgroup_variable", "subgroup_level", "driver"])
    return pd.concat(rows, ignore_index=True)

def run_analysis(
    df_num,
    df_raw,
    target,
    x_vars,
    sg_var,
    methods,
    include_bootstrap=False,
    bootstrap_resamples=DEFAULT_BOOTSTRAP_RESAMPLES,
):
    predictors = [c for c in (x_vars if x_vars else df_num.columns) if c != target and c in df_num.columns]
    if not predictors:
        return {"error": "No valid predictor columns available."}
    if not methods:
        return {"error": "Select at least one method."}

    model_df = df_num[[target, *predictors]].copy()
    if sg_var:
        if sg_var not in df_raw.columns:
            return {"error": f"Subgroup variable '{sg_var}' not found."}
        model_df[sg_var] = df_raw[sg_var]

    method_params = {
        "random_forest": {"n_estimators": 300, "n_repeats": 8},
        "xgboost": {"n_estimators": 150, "max_depth": 3},
        "shap": {"n_estimators": 150, "max_depth": 3},
    }
    bootstrap_methods = [method for method in methods if method in DEFAULT_BOOTSTRAP_METHODS] if include_bootstrap else None
    bootstrap_params = {
        "n_resamples": int(bootstrap_resamples),
        "random_state": 454,
        "min_valid_resamples": 8,
    } if include_bootstrap else None

    try:
        kda_result = run_kda(
            model_df,
            y_var=target,
            x_vars=predictors,
            methods=methods,
            subgroup=sg_var,
            method_params=method_params,
            bootstrap_methods=bootstrap_methods,
            bootstrap_params=bootstrap_params,
        )
    except Exception as exc:
        return {"error": str(exc)}

    driver_scores = _importance_to_series(kda_result.importance_table)
    export_table = _importance_export_table(kda_result)
    if not sg_var:
        return {
            "mode": "single",
            "target": target,
            "methods": methods,
            "ranked": driver_scores,
            "driver_scores": driver_scores,
            "top5": driver_scores.head(5),
            "export_table": export_table,
            "kda_result": kda_result,
            "warnings": kda_result.warnings,
            "include_bootstrap": include_bootstrap,
            "bootstrap_methods": bootstrap_methods or [],
            "bootstrap_resamples": int(bootstrap_resamples) if include_bootstrap else 0,
        }

    subgroup_results = []
    subgroup_summary = kda_result.subgroup_summary
    for group, subgroup_result in (kda_result.subgroup_results or {}).items():
        subgroup_ranked = _importance_to_series(subgroup_result.importance_table)
        subgroup_export_table = _importance_export_table(subgroup_result)
        subgroup_n = subgroup_result.diagnostics.loc[
            subgroup_result.diagnostics["metric"] == "rows_used",
            "value",
        ].iloc[0]
        subgroup_results.append(
            {
                "group": group,
                "n": subgroup_n,
                "skipped": False,
                "ranked": subgroup_ranked,
                "driver_scores": subgroup_ranked,
                "top5": subgroup_ranked.head(5),
                "export_table": subgroup_export_table,
                "kda_result": subgroup_result,
            }
        )
    if subgroup_summary is not None:
        existing = {str(item["group"]) for item in subgroup_results}
        for _, row in subgroup_summary.iterrows():
            level = str(row["subgroup_level"])
            if level in existing:
                continue
            subgroup_results.append(
                {
                    "group": level,
                    "n": int(row["rows_used"]),
                    "skipped": True,
                    "reason": row["reason"],
                    "ranked": pd.Series(dtype=float),
                    "driver_scores": pd.Series(dtype=float),
                    "top5": pd.Series(dtype=float),
                    "export_table": pd.DataFrame(),
                    "kda_result": None,
                }
            )
    return {
        "mode": "subgroup",
        "target": target,
        "methods": methods,
        "sg_var": sg_var,
        "results": subgroup_results,
        "subgroup_export_table": _combined_subgroup_export_table(kda_result, sg_var),
        "subgroup_summary": _subgroup_summary_table(kda_result),
        "warnings": kda_result.warnings,
        "kda_result": kda_result,
        "include_bootstrap": include_bootstrap,
        "bootstrap_methods": bootstrap_methods or [],
        "bootstrap_resamples": int(bootstrap_resamples) if include_bootstrap else 0,
    }

def render_dashboard():
    configure_page()
    st.markdown("""
    <div class="gbk-hero">
      <div class="gbk-eyebrow">GBK Toolbox</div>
      <h1>Marketing<br>Insights Suite</h1>
      <p>Upload a clean dataset, select your variables, choose a method, and run.<br>
      Pre-filter to relevant variables before uploading (1 Y &nbsp;·&nbsp; ~12 X &nbsp;·&nbsp; a few subgroup columns).</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="gbk-label">Upload dataset</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload .xlsx", type=["xlsx"], key="dashboard_upload", label_visibility="collapsed")

    if uploaded_file:
        base_signature = (uploaded_file.name, getattr(uploaded_file, "size", None))
        sheet_names = st.session_state.uploaded_sheet_names
        if st.session_state.uploaded_file_signature and st.session_state.uploaded_file_signature[:2] == base_signature:
            sheet_names = sheet_names or [st.session_state.uploaded_sheet_name]
        else:
            try:
                uploaded_file.seek(0)
                sheet_names = pd.ExcelFile(uploaded_file).sheet_names
                st.session_state.uploaded_sheet_names = sheet_names
            except Exception as e:
                sheet_names = None
                st.error(f"Error reading workbook sheets: {e}")

        selected_sheet = None
        if sheet_names:
            if len(sheet_names) > 1:
                selected_sheet = st.selectbox(
                    "Excel sheet",
                    sheet_names,
                    index=0,
                    key=f"sheet_select_{base_signature[0]}_{base_signature[1]}",
                )
            else:
                selected_sheet = sheet_names[0]
                st.markdown(
                    f'<div class="gbk-note" style="font-size:12px;color:rgba(255,255,255,0.50);">Sheet: <b>{selected_sheet}</b></div>',
                    unsafe_allow_html=True,
                )

        file_signature = (*base_signature, selected_sheet)
        if selected_sheet and st.session_state.uploaded_file_signature == file_signature:
            df_raw = st.session_state.uploaded_df_raw
            df_num = st.session_state.uploaded_df_num
            meta = st.session_state.uploaded_meta
        else:
            df_raw = df_num = meta = None
        try:
            if selected_sheet and (df_raw is None or df_num is None or meta is None):
                uploaded_file.seek(0)
                df_raw, df_num, meta = prepare_model_data(pd.read_excel(uploaded_file, sheet_name=selected_sheet))
                st.session_state.uploaded_df_raw = df_raw
                st.session_state.uploaded_df_num = df_num
                st.session_state.uploaded_meta = meta
                st.session_state.uploaded_filename = uploaded_file.name
                st.session_state.uploaded_sheet_name = selected_sheet
                st.session_state.uploaded_file_signature = file_signature
                st.session_state.analysis_result = None
        except Exception as e:
            st.error(f"Error loading file: {e}")

    df_raw = st.session_state.uploaded_df_raw
    df_num = st.session_state.uploaded_df_num
    meta = st.session_state.uploaded_meta

    if df_raw is None or df_num is None:
        st.markdown('<div class="gbk-note" style="color:rgba(255,255,255,0.25);">Upload a clean .xlsx dataset to begin.</div>', unsafe_allow_html=True)
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="gbk-card"><div class="gbk-card-kicker">File</div><div class="gbk-card-text" style="font-size:13px;">{st.session_state.uploaded_filename}</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="gbk-card"><div class="gbk-card-kicker">Respondents</div><div class="gbk-stat">{df_raw.shape[0]:,}</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="gbk-card"><div class="gbk-card-kicker">Total Columns</div><div class="gbk-stat">{df_raw.shape[1]:,}</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="gbk-card"><div class="gbk-card-kicker">Model-Ready</div><div class="gbk-stat">{df_num.shape[1]:,}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    y_options = meta.get("outcome_candidates") or df_num.columns.tolist()
    driver_candidates = meta.get("driver_candidates") or [c for c in df_num.columns if c not in y_options]
    compare_options = meta.get("subgroup_candidates") or []

    with st.expander("Advanced column setup"):
        st.markdown(
            '<div class="gbk-note">Use these controls when the automatic column detection needs a light correction. The backend still receives only the selected Y, X, and subgroup fields from the final setup below.</div>',
            unsafe_allow_html=True,
        )
        y_options = st.multiselect(
            "Outcome candidate columns",
            df_num.columns.tolist(),
            default=[c for c in y_options if c in df_num.columns],
            format_func=display_name,
            key="advanced_y_candidates",
        ) or y_options
        driver_candidates = st.multiselect(
            "Driver candidate columns",
            [c for c in df_num.columns if c not in y_options],
            default=[c for c in driver_candidates if c in df_num.columns and c not in y_options],
            format_func=display_name,
            key="advanced_x_candidates",
        ) or driver_candidates
        compare_options = st.multiselect(
            "Subgroup candidate columns",
            df_raw.columns.tolist(),
            default=[c for c in compare_options if c in df_raw.columns],
            format_func=lambda c: f"{display_name(c)} ({df_raw[c].nunique(dropna=True)} groups)",
            key="advanced_subgroup_candidates",
        ) or compare_options

    with st.container():

        # Step 1
        st.markdown('<div class="gbk-panel"><div class="gbk-panel-title">Step 1 · Outcome variable (Y)</div><div class="gbk-note">Select the business outcome you want to explain. If no likely outcome name is detected, all numeric columns are available here.</div></div>', unsafe_allow_html=True)
        y_var = st.selectbox("Y variable", ["(select)"] + y_options, format_func=lambda c: display_name(c) if c != "(select)" else "— select outcome —", label_visibility="collapsed", key="dash_y")
        y_selected = y_var if y_var != "(select)" else None

        st.markdown("<br>", unsafe_allow_html=True)

        # Step 2
        x_options = [c for c in driver_candidates if c != y_selected]
        n_x = len(x_options)
        if n_x > 30:
            x_hint = f'<div class="gbk-input-warning">⚠ {n_x} numeric columns detected. For cleaner KDA output, select the core driver variables instead of every numeric field.</div>'
        else:
            x_hint = f'<div style="font-size:11px;color:rgba(255,255,255,0.3);margin-top:5px;">{n_x} numeric variable{"s" if n_x!=1 else ""} available.</div>'

        st.markdown(
            f'<div class="gbk-panel"><div class="gbk-panel-title">Step 2 · Predictor variables (X)</div>'
            f'<div class="gbk-note">Choose the driver variables to compare. Leave empty to use all detected numeric driver columns.</div>'
            f'{x_hint}</div>',
            unsafe_allow_html=True
        )
        x_vars = st.multiselect("X variables", x_options, default=x_options[:12], format_func=display_name, label_visibility="collapsed", key="dash_x", placeholder="Detected brand image statements")

        st.markdown("<br>", unsafe_allow_html=True)

        # Step 3
        st.markdown('<div class="gbk-panel"><div class="gbk-panel-title">Step 3 · Compare results by group (optional)</div><div class="gbk-note">This repeats the same 1Y + 12X analysis within each selected group. Use <b>brand</b> for brand-by-brand drivers, or demographic fields such as Gender, Age Range, region, ecosystem, purchase channel, or drop frequency. Brand lookup fields like top brand and ownership brand are intentionally hidden.</div></div>', unsafe_allow_html=True)
        use_sg = st.checkbox("Compare results by group", key="dash_use_sg")
        sg_var = None
        if use_sg:
            if compare_options:
                sg_raw = st.selectbox("Compare by", ["(select)"] + compare_options, format_func=lambda c: f"{display_name(c)} ({df_raw[c].nunique(dropna=True)} groups)" if c != "(select)" else "— choose brand or segment —", label_visibility="collapsed", key="dash_sg")
                sg_var = sg_raw if sg_raw != "(select)" else None
            else:
                st.markdown('<div class="gbk-note">No suitable brand or segment columns detected.</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Step 4 — Method checkboxes with info boxes
        st.markdown(
            '<div class="gbk-panel"><div class="gbk-panel-title">Step 4 · Methods</div>'
            '<div class="gbk-note">Choose one or more driver-analysis methods. '
            'The backend will run selected methods and return a combined ranking.</div></div>',
            unsafe_allow_html=True
        )
        default_methods = set(DEFAULT_METHODS)
        selected_methods = []
        method_cols = st.columns(2)
        for idx, method_key in enumerate(ALL_METHODS):
            with method_cols[idx % 2]:
                checked = st.checkbox(
                    METHOD_LABELS.get(method_key, method_key),
                    value=method_key in default_methods,
                    key=f"dash_method_{method_key}",
                )
            if checked:
                selected_methods.append(method_key)
        for method_key in selected_methods:
            render_method_info_box(method_key)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            '<div class="gbk-panel"><div class="gbk-panel-title">Optional · Bootstrap confidence intervals</div>'
            '<div class="gbk-note">Leave this off for fast exploratory key driver analysis. '
            f'When enabled, the app resamples the data {DEFAULT_BOOTSTRAP_RESAMPLES} times by default and computes CI bands only for the lighter, more stable methods: '
            '<b>Correlation</b>, <b>Regression</b>, <b>Shapley / LMG</b>, and <b>Johnson Relative Weights</b>. '
            '<b>Random Forest</b>, <b>XGBoost</b>, and <b>SHAP</b> still appear as point estimates, but are not bootstrapped by default because each resample would refit tree ensembles and, for SHAP, rebuild explanations. That cost multiplies quickly across subgroups, so these methods stay point-only unless we add a deliberately slower advanced path.</div></div>',
            unsafe_allow_html=True,
        )
        include_bootstrap = st.checkbox("Calculate bootstrap confidence intervals", value=False, key="dash_bootstrap")
        bootstrap_resamples = DEFAULT_BOOTSTRAP_RESAMPLES
        selected_bootstrap_methods = [method for method in selected_methods if method in DEFAULT_BOOTSTRAP_METHODS]
        selected_heavy_methods = [method for method in selected_methods if method in HEAVY_BOOTSTRAP_METHODS]
        if include_bootstrap:
            with st.expander("Bootstrap controls"):
                bootstrap_resamples = st.number_input(
                    "Bootstrap resamples",
                    min_value=20,
                    max_value=300,
                    value=DEFAULT_BOOTSTRAP_RESAMPLES,
                    step=10,
                    help="More resamples make intervals smoother but slower. Subgroup runs multiply this cost by the number of included groups.",
                    key="bootstrap_resamples",
                )
            ci_method_label = ", ".join(METHOD_LABELS.get(method, method) for method in selected_bootstrap_methods) or "None"
            point_only_label = ", ".join(METHOD_LABELS.get(method, method) for method in selected_heavy_methods) or "None"
            st.markdown(
                f'<div class="gbk-note" style="margin-top:0.5rem;">'
                f'<b>CI will be calculated for:</b> {ci_method_label}<br>'
                f'<b>Point-only methods:</b> {point_only_label}'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Pipeline summary
        x_label = ", ".join(display_name(c) for c in x_vars) if x_vars else "All variables"
        sg_label = f"Compare by {display_name(sg_var)}" if use_sg and sg_var else "Overall only"
        y_label = display_name(y_selected) if y_selected else "Not selected"
        method_label = ", ".join(METHOD_LABELS.get(m, m) for m in selected_methods) if selected_methods else "None"
        ci_label = "On" if include_bootstrap else "Off"
        st.markdown(f"""
        <div class="gbk-panel" style="border-color:rgba(232,80,58,0.3);">
          <div class="gbk-panel-title">Pipeline summary</div>
          <div class="gbk-summary-grid">
            <div><div class="gbk-summary-key">Y</div><div class="gbk-summary-val">{y_label}</div></div>
            <div><div class="gbk-summary-key">X vars</div><div class="gbk-summary-val">{x_label}</div></div>
            <div><div class="gbk-summary-key">Subgroup</div><div class="gbk-summary-val">{sg_label}</div></div>
            <div><div class="gbk-summary-key">Methods / CI</div><div class="gbk-summary-val">{method_label}<br>Bootstrap: {ci_label}</div></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            run_clicked = st.button("Run Analysis", width="stretch")
        with btn_col2:
            clear_clicked = st.button("Clear Results", width="stretch")

    if clear_clicked:
        st.session_state.analysis_result = None

    if run_clicked:
        if not y_selected:
            st.error("Please select an outcome variable (Y).")
        else:
            with st.spinner(f"Running {method_label}..."):
                result = run_analysis(
                    df_num,
                    df_raw,
                    y_selected,
                    x_vars or None,
                    sg_var,
                    selected_methods,
                    include_bootstrap=include_bootstrap,
                    bootstrap_resamples=bootstrap_resamples,
                )
            st.session_state.analysis_result = result

    result = st.session_state.analysis_result

    if result:
        if "error" in result:
            st.error(result["error"])
        elif result["mode"] == "single":
            display_methods = st.multiselect(
                "Methods shown in chart",
                result["methods"],
                default=result["methods"],
                format_func=lambda method: METHOD_LABELS.get(method, method),
                key="single_chart_methods",
            )
            active_methods = display_methods or result["methods"]
            chart_x_domain = chart_range_control(result["kda_result"], active_methods, "single_chart")
            render_interval_chart(
                result["kda_result"],
                active_methods,
                title="Interactive driver comparison",
                chart_x_domain=chart_x_domain,
            )
            render_insights(result["target"], result["driver_scores"])
            render_next_steps(result["target"], result["driver_scores"])
            for warning in result.get("warnings", []):
                st.warning(warning)
            with st.expander("Actual score export table"):
                st.dataframe(result["export_table"], width="stretch")
                st.download_button(
                    "Download actual scores CSV",
                    result["export_table"].to_csv(index=False).encode("utf-8"),
                    file_name="driver_actual_scores.csv",
                    mime="text/csv",
                    width="stretch",
                )
            with st.expander("Ranking summary"):
                render_detail_table(result["driver_scores"])
                st.dataframe(result["kda_result"].ranking_table, width="stretch")
            with st.expander("Method diagnostics"):
                st.dataframe(result["kda_result"].diagnostics, width="stretch")
        elif result["mode"] == "subgroup":
            st.markdown(f'<div class="gbk-panel"><div class="gbk-panel-title">Compare by · {_auto_label(result["sg_var"])}</div><div class="gbk-note">Each section repeats the same selected Y and X variables within one <b>{_auto_label(result["sg_var"])}</b> group.</div></div>', unsafe_allow_html=True)
            display_methods = st.multiselect(
                "Methods shown in subgroup charts",
                result["methods"],
                default=result["methods"],
                format_func=lambda method: METHOD_LABELS.get(method, method),
                key="subgroup_chart_methods",
            )
            active_methods = display_methods or result["methods"]
            for warning in result.get("warnings", []):
                st.warning(warning)
            summary = result.get("subgroup_summary")
            if summary is not None and not summary.empty:
                included_n = int((summary["status"] == "included").sum())
                skipped_n = int((summary["status"] == "skipped").sum())
                st.markdown(
                    f'<div class="gbk-panel"><div class="gbk-panel-title">Subgroup run status</div>'
                    f'<div class="gbk-note"><b>{included_n}</b> groups included · <b>{skipped_n}</b> groups skipped. '
                    f'Skipped groups are excluded from charts and exports because they do not have enough complete rows for the selected Y/X setup.</div></div>',
                    unsafe_allow_html=True,
                )
                with st.expander("Subgroup status table"):
                    st.dataframe(summary, width="stretch")
            chart_x_domain = chart_range_control(result["kda_result"], active_methods, "subgroup_chart")
            for item in result["results"]:
                if item["skipped"]:
                    reason = item.get("reason") or "Not enough complete rows for the selected analysis."
                    st.markdown(
                        f'<div class="gbk-warning-card"><b>Skipped {item["group"]}</b><br>'
                        f'n={int(item["n"]):,}. {reason}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(f'<div style="font-size:13px;font-weight:700;color:#E8503A;margin:1rem 0 0.25rem;text-transform:uppercase;letter-spacing:1.5px;">{_auto_label(result["sg_var"])}: {item["group"]} · n={item["n"]:,}</div>', unsafe_allow_html=True)
                    render_interval_chart(
                        item["kda_result"],
                        active_methods,
                        title=f"All drivers — {item['group']}",
                        chart_x_domain=chart_x_domain,
                    )
                    st.dataframe(item["export_table"], width="stretch")
            with st.expander("Combined subgroup score export"):
                st.dataframe(result["subgroup_export_table"], width="stretch")
                st.download_button(
                    "Download subgroup actual scores CSV",
                    result["subgroup_export_table"].to_csv(index=False).encode("utf-8"),
                    file_name="subgroup_driver_actual_scores.csv",
                    mime="text/csv",
                    width="stretch",
                )
            with st.expander("Overall ranking summary"):
                st.dataframe(result["kda_result"].ranking_table, width="stretch")

    with st.expander("Raw data preview"):
        st.dataframe(df_raw.head(), width="stretch")

    with st.expander("Cleaning details"):
        st.markdown(f"""
        <div class="gbk-note">
          <b>Excluded (ID/date/meta):</b><br>{pill_tags(meta['excluded_cols'])}<br><br>
          <b>Dropped (high missing):</b><br>{pill_tags(meta['drop_missing_cols'])}<br><br>
          <b>Subgroup candidates:</b><br>{pill_tags(meta['subgroup_candidates'])}<br><br>
          <b>Dropped (constant):</b><br>{pill_tags(meta['constant_cols'])}
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    render_dashboard()
