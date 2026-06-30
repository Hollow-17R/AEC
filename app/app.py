"""
E-Sports Player Performance Classifier — Streamlit Dashboard
=============================================================
Premium, gaming-inspired dashboard with 5 interactive tabs for exploring
CS:GO player data, model performance, live predictions, and feature insights.
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure src package is importable
# ---------------------------------------------------------------------------
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import joblib

from src.utils import (
    PROJECT_ROOT,
    DATA_RAW,
    DATA_PROCESSED,
    MODELS_DIR,
    FIGURES_DIR,
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    BEST_MODEL_PATH,
    SCALER_PATH,
    FEATURE_COLUMNS_PATH,
    LABEL_ENCODER_PATH,
    ORIGINAL_FEATURES,
    ENGINEERED_FEATURES,
    MODEL_FEATURES,
    MODEL_NAMES,
    PERFORMANCE_CATEGORIES,
    PERFORMANCE_COLORS,
    CATEGORY_LABELS,
    TARGET_COLUMN,
)
from components import (
    inject_custom_css,
    render_metric_card,
    render_prediction_result,
    render_confidence_gauge,
    render_model_comparison_card,
    render_header,
    render_sidebar_info,
    PRIMARY,
    ACCENT,
    SUCCESS,
    WARNING,
    DANGER,
    BG_DARK,
    BG_CARD,
    TEXT_PRIMARY,
    TEXT_MUTED,
)

# ============================================================================
# Page Configuration (must be first Streamlit command)
# ============================================================================
st.set_page_config(
    page_title="E-Sports Performance Classifier",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject premium CSS
inject_custom_css()


# ============================================================================
# Cached Loaders
# ============================================================================
@st.cache_data(show_spinner=False)
def load_processed_data() -> pd.DataFrame:
    """Load the processed dataset."""
    if PROCESSED_DATA_PATH.exists():
        return pd.read_csv(PROCESSED_DATA_PATH)
    if RAW_DATA_PATH.exists():
        return pd.read_csv(RAW_DATA_PATH)
    return pd.DataFrame()


@st.cache_resource(show_spinner=False)
def load_best_model():
    """Load the best trained model."""
    if BEST_MODEL_PATH.exists():
        return joblib.load(BEST_MODEL_PATH)
    return None


@st.cache_resource(show_spinner=False)
def load_scaler():
    """Load the fitted scaler."""
    if SCALER_PATH.exists():
        return joblib.load(SCALER_PATH)
    return None


@st.cache_resource(show_spinner=False)
def load_label_encoder():
    """Load the fitted label encoder."""
    if LABEL_ENCODER_PATH.exists():
        return joblib.load(LABEL_ENCODER_PATH)
    return None


@st.cache_resource(show_spinner=False)
def load_feature_columns():
    """Load the ordered feature column list used during training."""
    if FEATURE_COLUMNS_PATH.exists():
        return joblib.load(FEATURE_COLUMNS_PATH)
    return MODEL_FEATURES


@st.cache_resource(show_spinner=False)
def load_model_by_name(name: str):
    """Load a specific model from models/ directory."""
    slug = name.lower().replace(" ", "_")
    path = MODELS_DIR / f"{slug}.pkl"
    if path.exists():
        return joblib.load(path)
    return None


@st.cache_data(show_spinner=False)
def load_model_metrics() -> dict:
    """Load model comparison metrics if saved as CSV/JSON."""
    csv_path = MODELS_DIR / "model_comparison.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        return df.to_dict("records")
    return []


# ============================================================================
# Helper: safe image display
# ============================================================================
def _show_image(filename: str, caption: str = ""):
    """Display an image from FIGURES_DIR; show info message if missing."""
    path = FIGURES_DIR / filename
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"📷 *{caption or filename}* will appear here after running the pipeline.")


# ============================================================================
# Helper: Plotly theming
# ============================================================================
_PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(26,26,46,0.5)",
    font=dict(color=TEXT_PRIMARY, family="Inter"),
    margin=dict(l=40, r=40, t=50, b=40),
    legend=dict(bgcolor="rgba(26,26,46,0.6)", bordercolor="#30304a", borderwidth=1),
)


# ============================================================================
# Sidebar
# ============================================================================
render_sidebar_info()

# Model selector in sidebar
st.sidebar.markdown("---")
selected_model_name = st.sidebar.selectbox(
    "🧠 Model for Prediction",
    options=MODEL_NAMES,
    index=1,  # default to Random Forest
    key="sidebar_model_select",
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div class="sidebar-section">
        <h4>ℹ️ ABOUT</h4>
        <p>Final-Year Engineering Project<br>
        Built with ❤️ using Python &amp; Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# Main Tabs
# ============================================================================
tab_overview, tab_eda, tab_models, tab_predict, tab_features = st.tabs(
    ["🏠 Overview", "📊 Exploratory Data Analysis", "🤖 Model Performance",
     "🎯 Predict Performance", "📈 Feature Insights"]
)

df = load_processed_data()

# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — Overview
# ─────────────────────────────────────────────────────────────────────────────
with tab_overview:
    render_header()

    # -- Key metric cards -------------------------------------------------- #
    total_players = len(df) if not df.empty else "—"
    num_features = len(MODEL_FEATURES)
    num_models = len(MODEL_NAMES)

    # Attempt to get best accuracy from saved metrics
    metrics_list = load_model_metrics()
    best_acc = "—"
    if metrics_list:
        try:
            accs = [m.get("accuracy", m.get("Accuracy", 0)) for m in metrics_list]
            best_acc = f"{max(accs) * 100:.1f}%"
        except Exception:
            pass

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Total Players", total_players, "in dataset", PRIMARY)
    with c2:
        render_metric_card("Features Used", num_features, "original + engineered", ACCENT)
    with c3:
        render_metric_card("Models Trained", num_models, "classifiers", WARNING)
    with c4:
        render_metric_card("Best Accuracy", best_acc, "top model", SUCCESS)

    st.markdown("<br>", unsafe_allow_html=True)

    # -- Dataset summary & distribution ------------------------------------ #
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown(
            '<div class="glass-card"><h3 class="gradient-text">📋 Dataset Summary</h3></div>',
            unsafe_allow_html=True,
        )
        if not df.empty:
            st.dataframe(
                df.describe().T.style.format("{:.2f}").set_properties(
                    **{"background-color": BG_CARD, "color": TEXT_PRIMARY}
                ),
                use_container_width=True,
                height=340,
            )
        else:
            st.info("Dataset not loaded yet. Run the preprocessing pipeline first.")

    with col_right:
        st.markdown(
            '<div class="glass-card"><h3 class="gradient-text">🎯 Performance Tier Distribution</h3></div>',
            unsafe_allow_html=True,
        )
        if not df.empty and TARGET_COLUMN in df.columns:
            tier_counts = df[TARGET_COLUMN].value_counts()
            fig_pie = go.Figure(
                go.Pie(
                    labels=tier_counts.index.tolist(),
                    values=tier_counts.values.tolist(),
                    hole=0.5,
                    marker=dict(
                        colors=[PERFORMANCE_COLORS.get(l, PRIMARY) for l in tier_counts.index],
                        line=dict(color=BG_DARK, width=2),
                    ),
                    textfont=dict(family="Inter", size=13),
                    hoverinfo="label+percent+value",
                )
            )
            fig_pie.update_layout(
                **_PLOTLY_LAYOUT,
                showlegend=True,
                height=340,
                title=None,
            )
            st.plotly_chart(fig_pie, use_container_width=True, key="overview_pie")
        else:
            st.info("Target column not found. Run the pipeline to generate performance tiers.")

    # -- Project description card ------------------------------------------ #
    st.markdown(
        """
        <div class="animated-border" style="margin-top:10px;">
            <div class="animated-border-inner">
                <h3 class="gradient-text">🕹️ About This Project</h3>
                <p style="font-family:'Inter',sans-serif;color:#8892b0;line-height:1.75;">
                    This dashboard showcases a <strong style="color:#e0e0e0;">machine-learning
                    pipeline</strong> that classifies CS:GO players into
                    <strong style="color:#00ff87;">High</strong>,
                    <strong style="color:#f0c000;">Medium</strong>, and
                    <strong style="color:#ff4757;">Low</strong> performance tiers based on
                    in-game statistics such as kills, deaths, headshot percentage, damage per
                    round, and more. Seven engineered features — including KDR, Impact Score,
                    and Consistency Score — enrich the raw stats for better classification
                    accuracy across Logistic Regression, Random Forest, and XGBoost models.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — Exploratory Data Analysis
# ─────────────────────────────────────────────────────────────────────────────
with tab_eda:
    st.markdown(
        '<div class="glass-card"><h3 class="gradient-text">📊 Exploratory Data Analysis</h3>'
        "<p style='color:#8892b0;font-family:Inter,sans-serif;font-size:0.92rem;'>"
        "Visual exploration of CS:GO player statistics.</p></div>",
        unsafe_allow_html=True,
    )

    eda_plots = [
        ("target_distribution.png", "Performance Tier Distribution — class balance across High / Medium / Low tiers"),
        ("feature_distributions.png", "Feature Distributions — histograms of all numeric features"),
        ("correlation_heatmap.png", "Correlation Heatmap — inter-feature Pearson correlations"),
        ("boxplots.png", "Box Plots — feature spread and outlier detection by performance tier"),
        ("scatter_analysis.png", "Scatter Analysis — pairwise relationships between key features"),
        ("violin_plots.png", "Violin Plots — distribution shape across performance categories"),
    ]

    for i in range(0, len(eda_plots), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(eda_plots):
                with col:
                    fname, caption = eda_plots[idx]
                    _show_image(fname, caption)

# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 — Model Performance
# ─────────────────────────────────────────────────────────────────────────────
with tab_models:
    st.markdown(
        '<div class="glass-card"><h3 class="gradient-text">🤖 Model Performance Comparison</h3>'
        "<p style='color:#8892b0;font-family:Inter,sans-serif;font-size:0.92rem;'>"
        "Side-by-side evaluation of all trained classifiers.</p></div>",
        unsafe_allow_html=True,
    )

    # -- Model comparison table -------------------------------------------- #
    if metrics_list:
        metrics_df = pd.DataFrame(metrics_list)
        st.markdown("#### 📋 Metrics Table")
        st.dataframe(
            metrics_df.style.format(
                {c: "{:.4f}" for c in metrics_df.select_dtypes("number").columns}
            ).highlight_max(
                subset=metrics_df.select_dtypes("number").columns, color="#00d2ff33"
            ),
            use_container_width=True,
        )
    else:
        st.info("Model comparison metrics will appear here after training. Run the model training pipeline first.")

    with st.expander("📐 **Metric Formulas & Definitions**", expanded=False):
        st.markdown(
            """
| Metric | Formula | Description |
|--------|---------|-------------|
| **Accuracy** | `(TP + TN) / (TP + TN + FP + FN)` | Proportion of all predictions that are correct |
| **Precision** | `TP / (TP + FP)` | Of all predicted positives, how many are actually positive |
| **Recall** | `TP / (TP + FN)` | Of all actual positives, how many are correctly identified |
| **F1 Score** | `2 × (Precision × Recall) / (Precision + Recall)` | Harmonic mean of Precision and Recall |

> **TP** = True Positives · **TN** = True Negatives · **FP** = False Positives · **FN** = False Negatives
>
> In the multi-class setting, per-class metrics are combined via **weighted average** (weighted by each class's support count).
            """
        )

    # -- Model cards ------------------------------------------------------- #
    if metrics_list:
        st.markdown("#### 🏆 Model Cards")
        card_cols = st.columns(len(metrics_list))
        for idx, record in enumerate(metrics_list):
            with card_cols[idx]:
                name = record.pop("model", record.pop("Model", f"Model {idx + 1}"))
                render_model_comparison_card(name, record)

    # -- Confusion matrices ------------------------------------------------ #
    st.markdown("#### 🔢 Confusion Matrices")
    cm_cols = st.columns(len(MODEL_NAMES))
    for idx, mname in enumerate(MODEL_NAMES):
        with cm_cols[idx]:
            slug = mname.lower().replace(" ", "_")
            _show_image(f"confusion_matrix_{slug}.png", f"{mname} — Confusion Matrix")

    # -- ROC curves -------------------------------------------------------- #
    st.markdown("#### 📈 ROC Curves")
    roc_cols = st.columns(len(MODEL_NAMES))
    for idx, mname in enumerate(MODEL_NAMES):
        with roc_cols[idx]:
            slug = mname.lower().replace(" ", "_")
            _show_image(f"roc_curve_{slug}.png", f"{mname} — ROC Curve")

    # -- Learning curves --------------------------------------------------- #
    st.markdown("#### 📉 Learning Curves")
    lc_cols = st.columns(len(MODEL_NAMES))
    for idx, mname in enumerate(MODEL_NAMES):
        with lc_cols[idx]:
            slug = mname.lower().replace(" ", "_")
            _show_image(f"learning_curve_{slug}.png", f"{mname} — Learning Curve")

    # -- Overall comparison chart ------------------------------------------ #
    st.markdown("#### 📊 Overall Model Comparison")
    _show_image("model_comparison.png", "Accuracy / F1 / Precision / Recall across models")

    # -- Best model highlight ---------------------------------------------- #
    if metrics_list:
        try:
            best = max(
                metrics_list,
                key=lambda m: m.get("accuracy", m.get("Accuracy", 0)),
            )
            best_name = best.get("model", best.get("Model", "Best Model"))
            best_acc_val = best.get("accuracy", best.get("Accuracy", 0))
            st.markdown(
                f"""
                <div class="animated-border" style="margin-top:14px;">
                    <div class="animated-border-inner" style="text-align:center;">
                        <h3 class="gradient-text">🏆 Best Model: {best_name}</h3>
                        <p style="font-family:'Orbitron',sans-serif;font-size:2.2rem;
                           color:#00ff87;text-shadow:0 0 20px #00ff8730;">
                            {best_acc_val*100:.2f}% Accuracy
                        </p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        except Exception:
            pass

# ─────────────────────────────────────────────────────────────────────────────
# Tab 4 — Predict Performance
# ─────────────────────────────────────────────────────────────────────────────
with tab_predict:
    st.markdown(
        '<div class="glass-card"><h3 class="gradient-text">🎯 Live Performance Prediction</h3>'
        "<p style='color:#8892b0;font-family:Inter,sans-serif;font-size:0.92rem;'>"
        "Enter player stats below and let the model predict the performance tier.</p></div>",
        unsafe_allow_html=True,
    )

    # -- Preset profiles --------------------------------------------------- #
    PRESETS = {
        "Custom": {},
        "Pro Star Player": dict(
            maps_played=350, rounds_played=10000, kills_per_round=0.85,
            deaths_per_round=0.55, assists_per_round=0.15, headshot_percentage=55,
            damage_per_round=90, rating=1.25, opening_kills=350,
            opening_deaths=180, clutch_wins=120,
        ),
        "Average Player": dict(
            maps_played=150, rounds_played=5000, kills_per_round=0.65,
            deaths_per_round=0.68, assists_per_round=0.12, headshot_percentage=42,
            damage_per_round=70, rating=1.00, opening_kills=150,
            opening_deaths=160, clutch_wins=40,
        ),
        "Beginner": dict(
            maps_played=30, rounds_played=800, kills_per_round=0.45,
            deaths_per_round=0.82, assists_per_round=0.08, headshot_percentage=28,
            damage_per_round=52, rating=0.70, opening_kills=20,
            opening_deaths=50, clutch_wins=3,
        ),
    }

    preset = st.selectbox(
        "⚡ Quick Preset", options=list(PRESETS.keys()), key="preset_select"
    )
    defaults = PRESETS[preset]

    with st.form("prediction_form"):
        st.markdown("##### 📝 Player Statistics")
        fc1, fc2, fc3 = st.columns(3)

        with fc1:
            maps_played = st.slider(
                "Maps Played", 1, 500, defaults.get("maps_played", 100), key="f_maps"
            )
            rounds_played = st.slider(
                "Rounds Played", 100, 15000, defaults.get("rounds_played", 3000), key="f_rounds"
            )
            kills_per_round = st.slider(
                "Kills / Round", 0.30, 1.50, defaults.get("kills_per_round", 0.65),
                step=0.01, key="f_kpr",
            )
            deaths_per_round = st.slider(
                "Deaths / Round", 0.30, 1.00, defaults.get("deaths_per_round", 0.65),
                step=0.01, key="f_dpr",
            )
        with fc2:
            assists_per_round = st.slider(
                "Assists / Round", 0.00, 0.50, defaults.get("assists_per_round", 0.12),
                step=0.01, key="f_apr",
            )
            headshot_percentage = st.slider(
                "Headshot %", 20, 80, defaults.get("headshot_percentage", 42), key="f_hs"
            )
            damage_per_round = st.slider(
                "Damage / Round", 40.0, 120.0, defaults.get("damage_per_round", 70.0),
                step=0.5, key="f_dpr2",
            )
            rating = st.slider(
                "Rating", 0.50, 1.60, defaults.get("rating", 1.00),
                step=0.01, key="f_rating",
            )
        with fc3:
            opening_kills = st.slider(
                "Opening Kills", 1, 500, defaults.get("opening_kills", 100), key="f_ok"
            )
            opening_deaths = st.slider(
                "Opening Deaths", 1, 500, defaults.get("opening_deaths", 100), key="f_od"
            )
            clutch_wins = st.slider(
                "Clutch Wins", 0, 200, defaults.get("clutch_wins", 30), key="f_cw"
            )

        submitted = st.form_submit_button("🚀  Predict Performance", use_container_width=True)

    if submitted:
        # Compute derived original features
        total_kills = int(kills_per_round * rounds_played)
        total_deaths = int(deaths_per_round * rounds_played)
        total_assists = int(assists_per_round * rounds_played)

        # Compute engineered features
        kdr = kills_per_round / max(deaths_per_round, 0.01)
        impact_score = (kills_per_round * 0.4 + damage_per_round / 100 * 0.3
                        + headshot_percentage / 100 * 0.15 + rating * 0.15)
        survival_rate = 1 - deaths_per_round
        performance_index = rating * kdr
        assist_contribution = assists_per_round / max(kills_per_round, 0.01)
        opening_duel_win_rate = opening_kills / max(opening_kills + opening_deaths, 1)
        consistency_score = rating / max(kdr, 0.01)

        input_dict = dict(
            maps_played=maps_played,
            rounds_played=rounds_played,
            total_kills=total_kills,
            total_deaths=total_deaths,
            total_assists=total_assists,
            headshot_percentage=headshot_percentage,
            kills_per_round=kills_per_round,
            deaths_per_round=deaths_per_round,
            assists_per_round=assists_per_round,
            damage_per_round=damage_per_round,
            rating=rating,
            opening_kills=opening_kills,
            opening_deaths=opening_deaths,
            clutch_wins=clutch_wins,
            kdr=kdr,
            impact_score=impact_score,
            survival_rate=survival_rate,
            performance_index=performance_index,
            assist_contribution=assist_contribution,
            opening_duel_win_rate=opening_duel_win_rate,
            consistency_score=consistency_score,
        )

        # Try to load the selected model (fall back to best model)
        model = load_model_by_name(selected_model_name) or load_best_model()
        scaler = load_scaler()
        le = load_label_encoder()
        feature_cols = load_feature_columns()

        if model is None:
            st.error("❌ No trained model found. Please run the training pipeline first.")
        else:
            try:
                input_df = pd.DataFrame([input_dict])
                # Align to trained feature order
                input_df = input_df.reindex(columns=feature_cols, fill_value=0)

                if scaler is not None:
                    input_scaled = scaler.transform(input_df)
                else:
                    input_scaled = input_df.values

                pred_encoded = model.predict(input_scaled)[0]

                # Decode label
                if le is not None:
                    pred_label = le.inverse_transform([pred_encoded])[0]
                else:
                    pred_label = PERFORMANCE_CATEGORIES.get(int(pred_encoded), str(pred_encoded))

                # Probabilities
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(input_scaled)[0]
                else:
                    proba = np.zeros(3)
                    proba[int(pred_encoded)] = 1.0

                if le is not None:
                    labels = le.classes_
                else:
                    labels = CATEGORY_LABELS

                prob_dict = {str(labels[i]): float(proba[i]) for i in range(len(labels))}
                confidence = float(max(proba))

                # --- Display results ---------------------------------------- #
                st.markdown("<br>", unsafe_allow_html=True)
                res_left, res_right = st.columns([3, 2])

                with res_left:
                    render_prediction_result(str(pred_label), confidence, prob_dict)

                with res_right:
                    render_confidence_gauge(confidence, "Prediction Confidence")

                    # Probability bar chart
                    bar_fig = go.Figure(
                        go.Bar(
                            x=list(prob_dict.values()),
                            y=list(prob_dict.keys()),
                            orientation="h",
                            marker_color=[
                                PERFORMANCE_COLORS.get(l, PRIMARY) for l in prob_dict
                            ],
                            text=[f"{v*100:.1f}%" for v in prob_dict.values()],
                            textposition="auto",
                            textfont=dict(family="Orbitron", size=13),
                        )
                    )
                    bar_fig.update_layout(
                        **_PLOTLY_LAYOUT,
                        height=220,
                        title=dict(
                            text="Class Probabilities",
                            font=dict(size=14, color=TEXT_MUTED),
                        ),
                        xaxis=dict(
                            range=[0, 1], tickformat=".0%",
                            gridcolor="#30304a", gridwidth=0.5,
                        ),
                        yaxis=dict(categoryorder="array", categoryarray=["Low", "Medium", "High"]),
                    )
                    st.plotly_chart(bar_fig, use_container_width=True, key="prob_bar")

                # -- Feature contribution (simplified) ----------------------- #
                st.markdown(
                    '<div class="glass-card"><h4 class="gradient-text">'
                    "🔍 Input Feature Profile</h4></div>",
                    unsafe_allow_html=True,
                )
                if not df.empty:
                    means = df[feature_cols].mean() if all(c in df.columns for c in feature_cols) else None
                    if means is not None:
                        ratios = {
                            feat: input_dict.get(feat, 0) / max(means[feat], 1e-9)
                            for feat in feature_cols
                        }
                        ratio_df = (
                            pd.DataFrame(
                                {"Feature": list(ratios.keys()), "Value vs Dataset Mean": list(ratios.values())}
                            )
                            .sort_values("Value vs Dataset Mean", ascending=True)
                        )
                        feat_fig = go.Figure(
                            go.Bar(
                                x=ratio_df["Value vs Dataset Mean"],
                                y=ratio_df["Feature"],
                                orientation="h",
                                marker=dict(
                                    color=ratio_df["Value vs Dataset Mean"],
                                    colorscale=[[0, DANGER], [0.5, WARNING], [1, SUCCESS]],
                                ),
                                text=[f"{v:.2f}×" for v in ratio_df["Value vs Dataset Mean"]],
                                textposition="auto",
                                textfont=dict(family="Inter", size=11),
                            )
                        )
                        feat_fig.update_layout(
                            **_PLOTLY_LAYOUT,
                            height=420,
                            title=dict(
                                text="Feature Values Relative to Dataset Mean (1.0× = average)",
                                font=dict(size=13, color=TEXT_MUTED),
                            ),
                            xaxis=dict(title="Ratio to Mean", gridcolor="#30304a"),
                        )
                        st.plotly_chart(feat_fig, use_container_width=True, key="feat_profile")
                    else:
                        st.info("Feature comparison requires the processed dataset.")
                else:
                    st.info("Load the dataset to see how your inputs compare to dataset averages.")

            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

# ─────────────────────────────────────────────────────────────────────────────
# Tab 5 — Feature Insights
# ─────────────────────────────────────────────────────────────────────────────
with tab_features:
    st.markdown(
        '<div class="glass-card"><h3 class="gradient-text">📈 Feature Insights</h3>'
        "<p style='color:#8892b0;font-family:Inter,sans-serif;font-size:0.92rem;'>"
        "Understanding which features drive player performance classification.</p></div>",
        unsafe_allow_html=True,
    )

    # -- SHAP plots -------------------------------------------------------- #
    st.markdown("#### 🧩 SHAP Explanations")
    shap_plots = [
        ("shap_summary.png", "SHAP Summary Plot — global feature importance across all predictions"),
        ("shap_bar.png", "SHAP Bar Plot — mean absolute SHAP values per feature"),
        ("shap_dependence.png", "SHAP Dependence Plot — interaction effects"),
    ]
    for fname, caption in shap_plots:
        _show_image(fname, caption)

    # -- Feature importance ------------------------------------------------ #
    st.markdown("#### 📊 Feature Importance")
    _show_image("feature_importance.png", "Feature Importance — ranked by model-specific importance scores")

    # -- Feature explanations ---------------------------------------------- #
    st.markdown(
        """
        <div class="glass-card">
            <h4 class="gradient-text">📖 Key Feature Descriptions</h4>
            <table style="width:100%;border-collapse:collapse;font-family:'Inter',sans-serif;font-size:0.88rem;">
                <tr style="border-bottom:1px solid #30304a;">
                    <td style="padding:10px;color:#00d2ff;font-weight:600;">rating</td>
                    <td style="padding:10px;color:#8892b0;">
                        Overall HLTV 2.0 rating — the single most holistic measure of player impact.</td>
                </tr>
                <tr style="border-bottom:1px solid #30304a;">
                    <td style="padding:10px;color:#00d2ff;font-weight:600;">kdr</td>
                    <td style="padding:10px;color:#8892b0;">
                        Kill-to-Death Ratio — measures frag efficiency and survivability.</td>
                </tr>
                <tr style="border-bottom:1px solid #30304a;">
                    <td style="padding:10px;color:#00d2ff;font-weight:600;">damage_per_round</td>
                    <td style="padding:10px;color:#8892b0;">
                        Average damage dealt per round — captures aggression and utility usage.</td>
                </tr>
                <tr style="border-bottom:1px solid #30304a;">
                    <td style="padding:10px;color:#00d2ff;font-weight:600;">impact_score</td>
                    <td style="padding:10px;color:#8892b0;">
                        Composite metric blending kills, damage, headshots, and rating for
                        overall impact.</td>
                </tr>
                <tr style="border-bottom:1px solid #30304a;">
                    <td style="padding:10px;color:#00d2ff;font-weight:600;">headshot_percentage</td>
                    <td style="padding:10px;color:#8892b0;">
                        Fraction of kills that are headshots — indicates aim precision.</td>
                </tr>
                <tr style="border-bottom:1px solid #30304a;">
                    <td style="padding:10px;color:#00d2ff;font-weight:600;">opening_duel_win_rate</td>
                    <td style="padding:10px;color:#8892b0;">
                        Win rate in first-contact duels — critical for entry fraggers.</td>
                </tr>
                <tr>
                    <td style="padding:10px;color:#00d2ff;font-weight:600;">consistency_score</td>
                    <td style="padding:10px;color:#8892b0;">
                        Rating normalised by KDR — reveals players who perform beyond
                        raw kill numbers.</td>
                </tr>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )
