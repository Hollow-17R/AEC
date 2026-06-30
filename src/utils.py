"""
Utility module for E-Sports Player Performance Classification.
Contains path constants, logging configuration, and shared helper functions.
"""

import os
import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================================
# Path Constants
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
APP_DIR = PROJECT_ROOT / "app"

# Raw and processed data file paths
RAW_DATA_PATH = DATA_RAW / "csgo_player_stats.csv"
PROCESSED_DATA_PATH = DATA_PROCESSED / "csgo_processed.csv"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.pkl"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"

# ============================================================================
# Create Directories
# ============================================================================
for dir_path in [DATA_RAW, DATA_PROCESSED, MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Logging Configuration
# ============================================================================
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level=logging.INFO):
    """Configure logging for the entire project."""
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
    )
    # Suppress verbose third-party loggers
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a named logger instance."""
    return logging.getLogger(name)


# ============================================================================
# Performance Category Configuration
# ============================================================================
PERFORMANCE_CATEGORIES = {
    0: "Low",
    1: "Medium",
    2: "High",
}

PERFORMANCE_COLORS = {
    "High": "#00ff87",     # Neon green
    "Medium": "#f0c000",   # Gold
    "Low": "#ff4757",      # Red
}

CATEGORY_LABELS = ["Low", "Medium", "High"]

# Quantile thresholds for target variable creation
LOW_QUANTILE = 0.30
HIGH_QUANTILE = 0.70

# ============================================================================
# Feature Configuration
# ============================================================================
# Original features from dataset
ORIGINAL_FEATURES = [
    "maps_played", "rounds_played", "total_kills", "total_deaths",
    "total_assists", "headshot_percentage", "kills_per_round",
    "deaths_per_round", "assists_per_round", "damage_per_round",
    "rating", "opening_kills", "opening_deaths", "clutch_wins",
]

# Engineered features
ENGINEERED_FEATURES = [
    "kdr", "impact_score", "survival_rate", "performance_index",
    "assist_contribution", "opening_duel_win_rate", "consistency_score",
]

# All numeric features used for modeling
MODEL_FEATURES = ORIGINAL_FEATURES + ENGINEERED_FEATURES

# Target column name
TARGET_COLUMN = "performance_category"
TARGET_COLUMN_ENCODED = "performance_encoded"

# ============================================================================
# Model Configuration
# ============================================================================
RANDOM_STATE = 42
TEST_SIZE = 0.20

MODEL_NAMES = ["Logistic Regression", "Random Forest", "XGBoost"]

# ============================================================================
# Helper Functions
# ============================================================================


def save_model(model, filepath: Path):
    """Save a trained model to disk."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)
    logger = get_logger("utils")
    logger.info(f"Model saved to {filepath}")


def load_model(filepath: Path):
    """Load a trained model from disk."""
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")
    model = joblib.load(filepath)
    logger = get_logger("utils")
    logger.info(f"Model loaded from {filepath}")
    return model


def save_dataframe(df: pd.DataFrame, filepath: Path):
    """Save a DataFrame to CSV."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    logger = get_logger("utils")
    logger.info(f"DataFrame saved to {filepath} ({len(df)} rows)")


def load_dataframe(filepath: Path) -> pd.DataFrame:
    """Load a DataFrame from CSV."""
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    df = pd.read_csv(filepath)
    logger = get_logger("utils")
    logger.info(f"DataFrame loaded from {filepath} ({len(df)} rows)")
    return df


def save_figure(fig, filename: str, dpi: int = 150):
    """Save a matplotlib figure to the reports/figures directory."""
    filepath = FIGURES_DIR / filename
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    logger = get_logger("utils")
    logger.info(f"Figure saved to {filepath}")


def set_plot_style():
    """Set consistent matplotlib/seaborn styling for all plots."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.style.use("dark_background")
    sns.set_theme(
        style="darkgrid",
        palette="viridis",
        font_scale=1.1,
        rc={
            "figure.figsize": (12, 7),
            "figure.facecolor": "#0e1117",
            "axes.facecolor": "#1a1a2e",
            "axes.edgecolor": "#30304a",
            "axes.labelcolor": "#e0e0e0",
            "text.color": "#e0e0e0",
            "xtick.color": "#b0b0b0",
            "ytick.color": "#b0b0b0",
            "grid.color": "#30304a",
            "grid.alpha": 0.3,
            "legend.facecolor": "#1a1a2e",
            "legend.edgecolor": "#30304a",
        },
    )


def print_section_header(title: str):
    """Print a formatted section header for console output."""
    width = 70
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)
