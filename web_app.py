from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from predict import (
    DEFAULT_FEATURE_VALUES,
    load_config,
    load_model_and_metadata,
    predict_dataframe,
    resolve_model_path,
)
from run_pipeline import execute_pipeline


st.set_page_config(
    page_title="Instacart ML Studio",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    :root {
      --ink-900: #101c3a;
      --ink-700: #253a63;
      --ink-500: #4d6288;
      --cream: #f8f5ee;
      --ivory: #fffdf7;
      --line: #d9d4c7;
      --accent: #ff7a1a;
      --accent-2: #1e8f76;
      --shadow: 0 14px 30px rgba(16, 28, 58, 0.08);
    }

    html, body, [class*="css"] {
      font-family: 'Sora', sans-serif;
      color: var(--ink-900);
    }

    [data-testid="stAppViewContainer"] section.main,
    [data-testid="stAppViewContainer"] section.main * {
      color: var(--ink-900);
    }

    .stApp {
      background:
        radial-gradient(circle at 16% 8%, rgba(255, 218, 182, 0.7) 0%, rgba(255, 218, 182, 0) 35%),
        radial-gradient(circle at 88% 3%, rgba(187, 246, 224, 0.6) 0%, rgba(187, 246, 224, 0) 32%),
        linear-gradient(180deg, #f8f5ee 0%, #f3f7f7 100%);
    }

    .block-container {
      max-width: 1200px;
      padding-top: 3.3rem;
      padding-bottom: 2rem;
    }

    [data-testid="stSidebar"] {
      background: linear-gradient(180deg, #16223f 0%, #1f2640 58%, #1f2b36 100%);
      border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    [data-testid="stSidebar"] * {
      color: #f4f6ff;
    }

    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] textarea,
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
      background-color: rgba(8, 15, 34, 0.8) !important;
      color: #f4f6ff !important;
      border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }

    [data-testid="stSidebar"] .stButton > button {
      width: 100%;
      border-radius: 12px;
      border: 1px solid rgba(255, 166, 77, 0.2);
      background: linear-gradient(90deg, #ff8a2b 0%, #ffb062 100%);
      color: #1d1f29;
      font-weight: 800;
      letter-spacing: 0.2px;
      transition: all .2s ease;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
      transform: translateY(-1px);
      box-shadow: 0 8px 18px rgba(255, 138, 43, 0.25);
    }

    .hero-shell {
      position: relative;
      overflow: hidden;
      border: 3px solid #ff8c2e;
      border-radius: 22px;
      background: linear-gradient(125deg, rgba(255, 245, 227, 0.95) 0%, rgba(255, 253, 247, 0.95) 45%, rgba(227, 247, 241, 0.95) 100%);
      box-shadow: 0 14px 30px rgba(16, 28, 58, 0.09), 0 0 0 2px rgba(255, 140, 46, 0.22), 0 0 0 1px rgba(255, 255, 255, 0.72) inset;
      padding: 1.5rem 1.8rem;
      margin-top: 1.15rem;
      margin-bottom: 1rem;
      animation: drift-in .55s ease-out;
    }

    .hero-shell::after {
      content: "";
      position: absolute;
      inset: 0;
      border-radius: 22px;
      pointer-events: none;
      box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.9);
    }

    .hero-topline {
      display: inline-block;
      font-size: 0.78rem;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: var(--ink-700);
      padding: 0.2rem 0.55rem;
      border: 1px solid rgba(37, 58, 99, 0.2);
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.55);
      margin-bottom: 0.6rem;
    }

    .hero-shell h1 {
      margin: 0;
      line-height: 1.05;
      font-size: clamp(1.9rem, 3.5vw, 3rem);
      color: var(--ink-900);
      letter-spacing: -0.02em;
    }

    .hero-shell p {
      margin-top: 0.65rem;
      margin-bottom: 1rem;
      color: var(--ink-700);
      max-width: 62ch;
      line-height: 1.45;
      font-size: 1rem;
    }

    .hero-tags {
      display: flex;
      flex-wrap: wrap;
      gap: 0.45rem;
    }

    .hero-tag {
      font-size: 0.78rem;
      font-weight: 700;
      color: #273b64;
      background: rgba(255, 255, 255, 0.82);
      border: 1px solid rgba(39, 59, 100, 0.16);
      border-radius: 999px;
      padding: 0.26rem 0.62rem;
    }

    .glass-card {
      border: 1px solid var(--line);
      border-radius: 16px;
      background: var(--ivory);
      box-shadow: var(--shadow);
      padding: 0.95rem 1rem;
      margin-bottom: 0.9rem;
      animation: rise .45s ease-out;
    }

    .metric-card {
      border: 1px solid var(--line);
      border-radius: 16px;
      background: linear-gradient(180deg, #fffefb 0%, #fff8ef 100%);
      box-shadow: 0 10px 24px rgba(16, 28, 58, 0.06);
      padding: 0.95rem 1rem;
      min-height: 120px;
    }

    .metric-label {
      color: var(--ink-500);
      font-size: 0.77rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-weight: 700;
    }

    .metric-value {
      margin-top: 0.35rem;
      color: var(--ink-900);
      font-size: clamp(1.35rem, 2.4vw, 2.05rem);
      font-weight: 800;
      letter-spacing: -0.03em;
    }

    .metric-note {
      margin-top: 0.3rem;
      color: var(--accent-2);
      font-size: 0.8rem;
      font-weight: 600;
    }

    .mono {
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.86rem;
      color: #2d3f66;
    }

    .result-pill {
      border-radius: 12px;
      padding: 0.68rem 0.82rem;
      border: 1px solid rgba(30, 143, 118, 0.25);
      background: linear-gradient(90deg, rgba(212, 252, 231, 0.95) 0%, rgba(232, 255, 244, 0.95) 100%);
      color: #174a3f;
      font-weight: 700;
      margin-bottom: 0.65rem;
    }

    .warn-pill {
      border-radius: 12px;
      padding: 0.68rem 0.82rem;
      border: 1px solid rgba(255, 122, 26, 0.25);
      background: linear-gradient(90deg, rgba(255, 239, 220, 0.95) 0%, rgba(255, 245, 233, 0.95) 100%);
      color: #8d4f11;
      font-weight: 700;
      margin-bottom: 0.65rem;
    }

    .stButton > button {
      border-radius: 12px;
      border: 1px solid rgba(255, 122, 26, 0.35);
      background: linear-gradient(90deg, #ff7a1a 0%, #ff9f43 100%);
      color: #1f2230;
      font-weight: 800;
      letter-spacing: 0.2px;
      transition: all .2s ease;
    }

    .stButton > button:hover {
      transform: translateY(-1px);
      box-shadow: 0 8px 18px rgba(255, 122, 26, 0.25);
    }

    [data-testid="stTabs"] [role="tablist"] {
      gap: 0.45rem;
      margin-bottom: 0.4rem;
    }

    [data-testid="stTabs"] [role="tab"] {
      background: rgba(255, 255, 255, 0.78);
      border: 1px solid #ddd4c0;
      border-radius: 12px;
      padding: 0.42rem 0.85rem;
      color: var(--ink-700);
      font-weight: 700;
    }

    [data-testid="stTabs"] [aria-selected="true"] {
      background: linear-gradient(90deg, #ffedda 0%, #fff7ec 100%);
      border-color: rgba(255, 122, 26, 0.35) !important;
      color: #8a4f1a !important;
    }

    [data-testid="stDataFrame"] {
      border: 1px solid var(--line);
      border-radius: 12px;
      overflow: hidden;
      box-shadow: 0 8px 20px rgba(16, 28, 58, 0.05);
    }

    [data-testid="stMetricValue"] {
      color: var(--ink-900);
      font-weight: 800;
    }

    [data-testid="stMetricLabel"] {
      color: var(--ink-500);
      font-weight: 700;
    }

    [data-testid="stProgressBar"] > div > div {
      background: linear-gradient(90deg, #21b788 0%, #62d2a7 100%);
    }

    @keyframes drift-in {
      0% { transform: translateY(8px); opacity: 0; }
      100% { transform: translateY(0); opacity: 1; }
    }

    @keyframes rise {
      0% { transform: translateY(6px); opacity: 0; }
      100% { transform: translateY(0); opacity: 1; }
    }

    @media (max-width: 900px) {
      .hero-shell {
        padding: 1.1rem;
        border-radius: 16px;
      }

      .metric-card {
        min-height: 102px;
      }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_hero():
    st.markdown(
        """
        <div class="hero-shell">
          <div class="hero-topline">Production-Like Demo</div>
          <h1>Instacart ML Studio</h1>
          <p>
            A polished control panel for your end-to-end pipeline:
            trigger training, inspect evaluation artifacts, and run inference in one place.
          </p>
          <div class="hero-tags">
            <div class="hero-tag">ETL</div>
            <div class="hero-tag">Feature Engineering</div>
            <div class="hero-tag">Model Training</div>
            <div class="hero-tag">Interactive Inference</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def read_metrics(metrics_path: str | Path):
    p = Path(metrics_path)
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _format_metric(value):
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def _format_bytes(num_bytes: int):
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{num_bytes} B"


def render_metric_tile(label: str, value: str, note: str):
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value}</div>
          <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metrics(config: dict):
    metrics_path = config.get("training", {}).get("metrics_output_path", "artifacts/metrics.json")
    metrics = read_metrics(metrics_path)
    if not metrics:
        st.info("No metrics found yet. Run training first.")
        return

    val_metrics = metrics.get("validation", {})
    test_metrics = metrics.get("test", {})

    roc_auc_for_ui = test_metrics.get("roc_auc", metrics.get("roc_auc"))
    pr_auc_for_ui = test_metrics.get("pr_auc", metrics.get("pr_auc"))
    f1_for_ui = test_metrics.get("f1_at_threshold", metrics.get("f1_at_threshold"))

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_metric_tile("Test ROC-AUC", _format_metric(roc_auc_for_ui), "Holdout generalization")
    with m2:
        render_metric_tile("Test PR-AUC", _format_metric(pr_auc_for_ui), "Positive class focus")
    with m3:
        render_metric_tile(
            "Test F1 @ Threshold",
            _format_metric(f1_for_ui),
            "Final holdout score",
        )
    with m4:
        threshold_source = metrics.get("threshold_source", "unknown")
        render_metric_tile("Decision Threshold", _format_metric(metrics.get("threshold")), f"Selected on {threshold_source}")

    c1, c2, c3 = st.columns(3)
    c1.markdown(
        f"""
        <div class="glass-card">
          <div class="metric-label">Split Strategy</div>
          <div class="mono">{metrics.get("split_strategy", "N/A")}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c2.markdown(
        f"""
        <div class="glass-card">
          <div class="metric-label">Train Rows</div>
          <div class="mono">{metrics.get("train_rows", "N/A")}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c3.markdown(
        f"""
        <div class="glass-card">
          <div class="metric-label">Test Rows</div>
          <div class="mono">{metrics.get("test_rows", "N/A")}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    leakage = metrics.get("leakage_checks", {})
    overlap_msg = (
        f"train∩val={leakage.get('user_overlap_train_val', 'N/A')}, "
        f"train∩test={leakage.get('user_overlap_train_test', 'N/A')}, "
        f"val∩test={leakage.get('user_overlap_val_test', 'N/A')}"
    )
    st.markdown(
        f"""
        <div class="glass-card">
          <div class="metric-label">Leakage Checks (By user_id)</div>
          <div class="mono">{overlap_msg}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if val_metrics:
        st.markdown(
            f"""
            <div class="glass-card">
              <div class="metric-label">Validation Metrics (Threshold Tuning Split)</div>
              <div class="mono">roc_auc={_format_metric(val_metrics.get("roc_auc"))}, pr_auc={_format_metric(val_metrics.get("pr_auc"))}, f1={_format_metric(val_metrics.get("f1_at_threshold"))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    report = metrics.get("classification_report", {})
    if report:
        rows = []
        for label in ["0", "1", "macro avg", "weighted avg"]:
            if label in report:
                row = report[label]
                rows.append(
                    {
                        "label": label,
                        "precision": round(float(row.get("precision", 0)), 4),
                        "recall": round(float(row.get("recall", 0)), 4),
                        "f1-score": round(float(row.get("f1-score", 0)), 4),
                        "support": int(row.get("support", 0)),
                    }
                )
        if rows:
            st.markdown("### Classification Report")
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

    fi_path = Path(
        config.get("training", {}).get("feature_importance_output_path", "artifacts/feature_importance.csv")
    )
    if fi_path.exists():
        st.markdown("### Feature Importance")
        fi = pd.read_csv(fi_path)
        st.dataframe(fi, use_container_width=True)
        st.bar_chart(fi.set_index("feature")["importance"])


def render_prediction_playground(config_path: str):
    try:
        model, features, threshold, model_path = load_model_and_metadata(config_path, None)
    except Exception as e:
        st.warning(str(e))
        return

    st.markdown(
        f"""
        <div class="glass-card">
          <div class="metric-label">Loaded Model</div>
          <div class="mono">{model_path}</div>
          <div class="metric-label" style="margin-top:8px;">Decision Threshold</div>
          <div class="mono">{threshold:.4f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Interactive Feature Inputs")
    cols = st.columns(3)
    row_payload = {}
    for idx, feature in enumerate(features):
        default_value = float(DEFAULT_FEATURE_VALUES.get(feature, 0.0))
        with cols[idx % 3]:
            row_payload[feature] = st.number_input(
                feature,
                value=default_value,
                step=0.1,
                format="%.4f",
                key=f"input_{feature}",
            )

    if st.button("Run Inference", key="predict_btn"):
        input_df = pd.DataFrame([row_payload])
        pred_df = predict_dataframe(input_df, model, features, threshold)
        pred = int(pred_df.loc[0, "prediction"])
        prob = float(pred_df.loc[0, "probability"])
        status = "REORDERED (1)" if pred == 1 else "NOT REORDERED (0)"
        if pred == 1:
            st.markdown(
                f'<div class="result-pill">Prediction: {status} | Probability: {prob:.2%}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="warn-pill">Prediction: {status} | Probability: {prob:.2%}</div>',
                unsafe_allow_html=True,
            )
        st.progress(min(max(prob, 0.0), 1.0))
        st.dataframe(pred_df, use_container_width=True)


def render_artifacts(config: dict):
    st.markdown("### Pipeline Artifacts")
    model_path = Path(resolve_model_path(config))
    metrics_path = Path(config.get("training", {}).get("metrics_output_path", "artifacts/metrics.json"))
    report_path = Path(config.get("training", {}).get("report_output_path", "artifacts/classification_report.txt"))
    fi_path = Path(
        config.get("training", {}).get("feature_importance_output_path", "artifacts/feature_importance.csv")
    )
    db_path = Path("instacart.db")

    artifacts = [
        ("Model", model_path),
        ("Metrics", metrics_path),
        ("Report", report_path),
        ("Feature Importance", fi_path),
        ("SQLite", db_path),
    ]

    rows = []
    for name, path in artifacts:
        rows.append(
            {
                "artifact": name,
                "path": str(path),
                "status": "ready" if path.exists() else "missing",
                "size": _format_bytes(path.stat().st_size) if path.exists() else "0 B",
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


def main():
    render_hero()

    st.sidebar.header("Pipeline Controls")
    st.sidebar.caption("Run training and refresh artifacts from one place.")
    config_path = st.sidebar.text_input("Config Path", value="config/model_config.yaml")
    use_sample = st.sidebar.checkbox("Use Sample Data If Missing", value=True)
    sample_seed = st.sidebar.number_input("Sample Seed", min_value=1, max_value=99999, value=42, step=1)
    data_dir = st.sidebar.text_input("Custom Data Dir (optional)", value="")

    if st.sidebar.button("Train / Refresh Pipeline", key="train_btn"):
        with st.spinner("Running ETL + training pipeline..."):
            try:
                summary = execute_pipeline(
                    config_path=config_path,
                    data_dir=data_dir or None,
                    use_sample_data_if_missing=use_sample,
                    sample_seed=int(sample_seed),
                )
                st.sidebar.success("Pipeline completed.")
                st.session_state["pipeline_summary"] = summary
            except Exception as e:
                st.sidebar.error(str(e))

    summary = st.session_state.get("pipeline_summary")
    if summary:
        st.markdown(
            f"""
            <div class="glass-card">
              <div class="metric-label">Latest Run</div>
              <span class="mono">train_shape={summary.get("train_shape")}</span><br/>
              <span class="mono">feature_count={summary.get("feature_count")}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    try:
        config = load_config(config_path)
    except Exception as e:
        st.error(str(e))
        return

    tab1, tab2, tab3 = st.tabs(["Training Dashboard", "Prediction Playground", "Artifacts"])
    with tab1:
        render_metrics(config)
    with tab2:
        render_prediction_playground(config_path)
    with tab3:
        render_artifacts(config)


if __name__ == "__main__":
    main()
