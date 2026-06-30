"""
Reusable Streamlit UI components with premium gaming-inspired styling.
Provides glassmorphism cards, animated gradients, neon glow effects,
and a cohesive dark theme for the E-Sports Performance Classifier dashboard.
"""

import streamlit as st
import plotly.graph_objects as go

# ============================================================================
# Color Palette
# ============================================================================
PRIMARY = "#00d2ff"       # Cyan
ACCENT = "#ff00e5"        # Magenta
SUCCESS = "#00ff87"       # Neon green
WARNING = "#f0c000"       # Gold
DANGER = "#ff4757"        # Red
BG_DARK = "#0e1117"       # Main background
BG_CARD = "#1a1a2e"       # Card background
BORDER = "#30304a"        # Subtle borders
TEXT_PRIMARY = "#e0e0e0"
TEXT_MUTED = "#8892b0"


# ============================================================================
# Custom CSS Injection
# ============================================================================
def inject_custom_css():
    """Inject premium custom CSS: glassmorphism, neon glow, animated gradients."""
    st.markdown(
        """
        <style>
        /* ── Import font ─────────────────────────────────────── */
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;500;600;700&display=swap');

        /* ── Global overrides ────────────────────────────────── */
        .stApp {
            background: linear-gradient(135deg, #0e1117 0%, #13111c 40%, #0e1117 100%);
        }

        /* ── Custom scrollbar ────────────────────────────────── */
        ::-webkit-scrollbar { width: 8px; height: 8px; }
        ::-webkit-scrollbar-track { background: #0e1117; }
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(180deg, #00d2ff, #ff00e5);
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover { background: #00d2ff; }

        /* ── Glassmorphism card base ─────────────────────────── */
        .glass-card {
            background: rgba(26, 26, 46, 0.65);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(48, 48, 74, 0.6);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 16px;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .glass-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 32px rgba(0, 210, 255, 0.12);
        }

        /* ── Metric card ─────────────────────────────────────── */
        .metric-card {
            background: rgba(26, 26, 46, 0.7);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(48, 48, 74, 0.5);
            border-radius: 14px;
            padding: 22px 20px;
            text-align: center;
            position: relative;
            overflow: hidden;
            transition: all 0.35s ease;
        }
        .metric-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 3px;
            border-radius: 14px 14px 0 0;
        }
        .metric-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 40px rgba(0, 210, 255, 0.15);
            border-color: rgba(0, 210, 255, 0.3);
        }
        .metric-title {
            font-family: 'Inter', sans-serif;
            font-size: 0.82rem;
            font-weight: 500;
            color: #8892b0;
            text-transform: uppercase;
            letter-spacing: 1.2px;
            margin-bottom: 8px;
        }
        .metric-value {
            font-family: 'Orbitron', sans-serif;
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 4px;
            text-shadow: 0 0 20px;
        }
        .metric-subtitle {
            font-family: 'Inter', sans-serif;
            font-size: 0.78rem;
            color: #8892b0;
        }

        /* ── Neon glow pulse animation ───────────────────────── */
        @keyframes glowPulse {
            0%, 100% { box-shadow: 0 0 5px rgba(0,210,255,0.15), 0 0 20px rgba(0,210,255,0.05); }
            50%      { box-shadow: 0 0 10px rgba(0,210,255,0.25), 0 0 40px rgba(0,210,255,0.1); }
        }
        .neon-glow { animation: glowPulse 3s ease-in-out infinite; }

        /* ── Gradient text ───────────────────────────────────── */
        .gradient-text {
            background: linear-gradient(135deg, #00d2ff, #ff00e5);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        /* ── Header styling ──────────────────────────────────── */
        .app-header {
            text-align: center;
            padding: 30px 10px 20px;
            margin-bottom: 10px;
        }
        .app-header h1 {
            font-family: 'Orbitron', sans-serif;
            font-size: 2.5rem;
            font-weight: 900;
            background: linear-gradient(135deg, #00d2ff 0%, #ff00e5 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 6px;
            letter-spacing: 1px;
        }
        .app-header .subtitle {
            font-family: 'Inter', sans-serif;
            color: #8892b0;
            font-size: 1.05rem;
            font-weight: 300;
        }
        .header-line {
            width: 120px;
            height: 3px;
            background: linear-gradient(90deg, #00d2ff, #ff00e5);
            margin: 14px auto 0;
            border-radius: 2px;
        }

        /* ── Prediction result card ──────────────────────────── */
        .prediction-card {
            background: rgba(26, 26, 46, 0.75);
            backdrop-filter: blur(16px);
            border-radius: 18px;
            padding: 32px;
            text-align: center;
            position: relative;
            overflow: hidden;
        }
        .prediction-label {
            font-family: 'Orbitron', sans-serif;
            font-size: 2.8rem;
            font-weight: 900;
            text-shadow: 0 0 30px;
            margin-bottom: 10px;
        }
        .confidence-bar-bg {
            width: 100%;
            height: 10px;
            background: rgba(48, 48, 74, 0.6);
            border-radius: 5px;
            margin: 14px 0;
            overflow: hidden;
        }
        .confidence-bar-fill {
            height: 100%;
            border-radius: 5px;
            transition: width 1.2s ease;
        }

        /* ── Probability breakdown ───────────────────────────── */
        .prob-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid rgba(48,48,74,0.4);
        }
        .prob-label { font-family: 'Inter', sans-serif; font-weight: 500; font-size: 0.95rem; }
        .prob-value { font-family: 'Orbitron', sans-serif; font-weight: 700; font-size: 1rem; }

        /* ── Model comparison card ───────────────────────────── */
        .model-card {
            background: rgba(26, 26, 46, 0.65);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(48, 48, 74, 0.5);
            border-radius: 14px;
            padding: 20px;
            margin-bottom: 12px;
        }
        .model-card h3 {
            font-family: 'Orbitron', sans-serif;
            font-size: 1.1rem;
            margin-bottom: 12px;
        }
        .model-metric-row {
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
            font-family: 'Inter', sans-serif;
            font-size: 0.9rem;
            border-bottom: 1px solid rgba(48,48,74,0.3);
        }
        .model-metric-label { color: #8892b0; }
        .model-metric-val { color: #00d2ff; font-weight: 600; }

        /* ── Sidebar styling ─────────────────────────────────── */
        .sidebar-section {
            background: rgba(26, 26, 46, 0.5);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 14px;
            border: 1px solid rgba(48, 48, 74, 0.4);
        }
        .sidebar-section h4 {
            font-family: 'Orbitron', sans-serif;
            font-size: 0.85rem;
            color: #00d2ff;
            margin-bottom: 10px;
        }
        .sidebar-section p, .sidebar-section li {
            font-family: 'Inter', sans-serif;
            font-size: 0.82rem;
            color: #8892b0;
            line-height: 1.6;
        }

        /* ── Tab styling overrides ───────────────────────────── */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background: rgba(26, 26, 46, 0.4);
            border-radius: 12px;
            padding: 4px;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            font-family: 'Inter', sans-serif;
            font-weight: 500;
            padding: 10px 20px;
        }
        .stTabs [aria-selected="true"] {
            background: rgba(0, 210, 255, 0.12) !important;
            border-bottom: 2px solid #00d2ff !important;
        }

        /* ── Styled DataFrame ────────────────────────────────── */
        .stDataFrame {
            border-radius: 12px;
            overflow: hidden;
        }

        /* ── Animated gradient border for featured cards ────── */
        @keyframes gradientShift {
            0%   { background-position: 0% 50%; }
            50%  { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        .animated-border {
            position: relative;
            border-radius: 18px;
            padding: 2px;
            background: linear-gradient(135deg, #00d2ff, #ff00e5, #00d2ff);
            background-size: 200% 200%;
            animation: gradientShift 4s ease infinite;
        }
        .animated-border-inner {
            background: #0e1117;
            border-radius: 16px;
            padding: 24px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# Metric Card
# ============================================================================
def render_metric_card(title: str, value, subtitle: str = "", color: str = PRIMARY):
    """Render a premium glassmorphism metric card."""
    st.markdown(
        f"""
        <div class="metric-card neon-glow" style="border-top: 3px solid {color};">
            <div class="metric-title">{title}</div>
            <div class="metric-value" style="color:{color}; text-shadow: 0 0 20px {color}40;">
                {value}
            </div>
            <div class="metric-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# Prediction Result
# ============================================================================
_PERF_COLORS = {"High": SUCCESS, "Medium": WARNING, "Low": DANGER}


def render_prediction_result(prediction: str, confidence: float, probabilities: dict):
    """
    Render a large, animated prediction result card.

    Parameters
    ----------
    prediction : str
        Predicted class label ("High", "Medium", or "Low").
    confidence : float
        Confidence score in [0, 1].
    probabilities : dict
        Mapping of class label → probability (float 0-1).
    """
    color = _PERF_COLORS.get(prediction, PRIMARY)
    conf_pct = confidence * 100

    # Build probability rows
    prob_html = ""
    for label in ["High", "Medium", "Low"]:
        prob = probabilities.get(label, 0.0) * 100
        lbl_color = _PERF_COLORS.get(label, TEXT_PRIMARY)
        prob_html += (
            f'<div class="prob-row">'
            f'  <span class="prob-label" style="color:{lbl_color};">● {label}</span>'
            f'  <span class="prob-value" style="color:{lbl_color};">{prob:.1f}%</span>'
            f"</div>"
        )

    st.markdown(
        f"""
        <div class="prediction-card neon-glow" style="border: 2px solid {color}40;">
            <p style="font-family:'Inter',sans-serif;color:#8892b0;font-size:0.9rem;
               margin-bottom:4px;text-transform:uppercase;letter-spacing:2px;">
                Predicted Performance Tier
            </p>
            <div class="prediction-label" style="color:{color};text-shadow:0 0 30px {color}50;">
                {prediction.upper()}
            </div>
            <p style="font-family:'Inter',sans-serif;color:{TEXT_MUTED};font-size:0.95rem;">
                Confidence: <strong style="color:{color};">{conf_pct:.1f}%</strong>
            </p>
            <div class="confidence-bar-bg">
                <div class="confidence-bar-fill"
                     style="width:{conf_pct}%;background:linear-gradient(90deg,{color},{color}cc);">
                </div>
            </div>
            <div style="margin-top:20px;text-align:left;">
                <p style="font-family:'Inter',sans-serif;font-weight:600;color:{TEXT_PRIMARY};
                   margin-bottom:8px;font-size:0.9rem;">CLASS PROBABILITIES</p>
                {prob_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# Confidence Gauge
# ============================================================================
def render_confidence_gauge(confidence: float, label: str = "Confidence"):
    """Render a circular gauge dial using Plotly."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            number={"suffix": "%", "font": {"size": 36, "color": PRIMARY, "family": "Orbitron"}},
            title={"text": label, "font": {"size": 14, "color": TEXT_MUTED, "family": "Inter"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": BORDER},
                "bar": {"color": PRIMARY},
                "bgcolor": BG_CARD,
                "borderwidth": 1,
                "bordercolor": BORDER,
                "steps": [
                    {"range": [0, 40], "color": "rgba(255,71,87,0.15)"},
                    {"range": [40, 70], "color": "rgba(240,192,0,0.15)"},
                    {"range": [70, 100], "color": "rgba(0,255,135,0.15)"},
                ],
                "threshold": {
                    "line": {"color": ACCENT, "width": 3},
                    "thickness": 0.8,
                    "value": confidence * 100,
                },
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": TEXT_PRIMARY},
        height=250,
        margin=dict(l=30, r=30, t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True, key=f"gauge_{label.replace(' ', '_')}")


# ============================================================================
# Model Comparison Card
# ============================================================================
def render_model_comparison_card(model_name: str, metrics_dict: dict):
    """Render a styled card showing model name and key metrics."""
    rows = ""
    for metric, value in metrics_dict.items():
        formatted = f"{value:.4f}" if isinstance(value, float) else str(value)
        rows += (
            f'<div class="model-metric-row">'
            f'  <span class="model-metric-label">{metric}</span>'
            f'  <span class="model-metric-val">{formatted}</span>'
            f"</div>"
        )

    st.markdown(
        f"""
        <div class="model-card neon-glow">
            <h3 class="gradient-text">🤖 {model_name}</h3>
            {rows}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# Header
# ============================================================================
def render_header():
    """Render the premium app header with title and decorative element."""
    st.markdown(
        """
        <div class="app-header">
            <h1>🎮 E-Sports Player Performance Classifier</h1>
            <p class="subtitle">
                Machine-Learning-Powered CS:GO Player Tier Prediction
            </p>
            <div class="header-line"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# Sidebar Info
# ============================================================================
def render_sidebar_info():
    """Render sidebar with project information and links."""
    st.sidebar.markdown(
        """
        <div class="sidebar-section">
            <h4>🎮 PROJECT INFO</h4>
            <p>
                <strong style="color:#e0e0e0;">E-Sports Player Performance Classification</strong><br>
                Predicts CS:GO player performance tiers
                (<span style="color:#00ff87;">High</span> ·
                 <span style="color:#f0c000;">Medium</span> ·
                 <span style="color:#ff4757;">Low</span>)
                using match statistics and engineered features.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown(
        """
        <div class="sidebar-section">
            <h4>🤖 MODELS</h4>
            <ul>
                <li>Logistic Regression</li>
                <li>Random Forest</li>
                <li>XGBoost</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown(
        """
        <div class="sidebar-section">
            <h4>📊 FEATURES</h4>
            <p>14 original features + 7 engineered features including KDR,
            Impact Score, Survival Rate, and Consistency Score.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown(
        f"""
        <div class="sidebar-section">
            <h4>⚡ TECH STACK</h4>
            <p>Python · Scikit-learn · XGBoost · Pandas<br>
            Streamlit · Plotly · SHAP</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
