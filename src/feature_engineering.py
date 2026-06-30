"""
Feature Engineering for E-Sports Player Performance Classification.

Derives seven engineered features and a three-class target variable
(Low / Medium / High) from the preprocessed CS:GO player data.
"""

import numpy as np
import pandas as pd

from src.utils import (
    HIGH_QUANTILE,
    LOW_QUANTILE,
    PERFORMANCE_CATEGORIES,
    PROCESSED_DATA_PATH,
    TARGET_COLUMN,
    TARGET_COLUMN_ENCODED,
    get_logger,
    load_dataframe,
    save_dataframe,
    setup_logging,
)

logger = get_logger(__name__)

# Output path — enriched dataset overwrites processed file
ENRICHED_DATA_PATH = PROCESSED_DATA_PATH


# ============================================================================
# Individual feature computations
# ============================================================================

def compute_kdr(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Kill-Death Ratio (KDR).

    ``KDR = total_kills / total_deaths``

    Zero deaths are replaced with 1 to avoid division by zero.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        DataFrame with an added ``kdr`` column.
    """
    df = df.copy()
    safe_deaths = df["total_deaths"].replace(0, 1)
    df["kdr"] = np.round(df["total_kills"] / safe_deaths, 4)
    logger.info("Computed 'kdr' — mean=%.3f, std=%.3f", df["kdr"].mean(), df["kdr"].std())
    return df


def compute_impact_score(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Impact Score.

    ``impact = 0.30·kpr + 0.20·(dpr/100) + 0.25·(ok/maps) + 0.15·(cw/maps) + 0.10·(1 − dpr_death)``

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        DataFrame with an added ``impact_score`` column.
    """
    df = df.copy()
    maps = df["maps_played"].replace(0, 1)

    impact = (
        0.30 * df["kills_per_round"]
        + 0.20 * (df["damage_per_round"] / 100.0)
        + 0.25 * (df["opening_kills"] / maps)
        + 0.15 * (df["clutch_wins"] / maps)
        + 0.10 * (1.0 - df["deaths_per_round"])
    )
    df["impact_score"] = np.round(impact, 4)
    logger.info("Computed 'impact_score' — mean=%.3f, std=%.3f", df["impact_score"].mean(), df["impact_score"].std())
    return df


def compute_survival_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Survival Rate.

    ``survival_rate = 1 − (total_deaths / rounds_played)``

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        DataFrame with an added ``survival_rate`` column.
    """
    df = df.copy()
    safe_rounds = df["rounds_played"].replace(0, 1)
    df["survival_rate"] = np.round(1.0 - (df["total_deaths"] / safe_rounds), 4)
    logger.info("Computed 'survival_rate' — mean=%.3f, std=%.3f", df["survival_rate"].mean(), df["survival_rate"].std())
    return df


def compute_assist_contribution(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Assist Contribution.

    ``assist_contribution = apr / (kpr + apr)``

    When both ``kpr`` and ``apr`` are zero the result is set to 0.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        DataFrame with an added ``assist_contribution`` column.
    """
    df = df.copy()
    denominator = df["kills_per_round"] + df["assists_per_round"]
    df["assist_contribution"] = np.where(
        denominator == 0,
        0.0,
        df["assists_per_round"] / denominator,
    )
    df["assist_contribution"] = np.round(df["assist_contribution"], 4)
    logger.info(
        "Computed 'assist_contribution' — mean=%.3f, std=%.3f",
        df["assist_contribution"].mean(), df["assist_contribution"].std(),
    )
    return df


def compute_opening_duel_win_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Opening Duel Win Rate.

    ``opening_duel_win_rate = opening_kills / (opening_kills + opening_deaths)``

    Zero total opening duels produce a win rate of 0.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        DataFrame with an added ``opening_duel_win_rate`` column.
    """
    df = df.copy()
    total_duels = df["opening_kills"] + df["opening_deaths"]
    df["opening_duel_win_rate"] = np.where(
        total_duels == 0,
        0.0,
        df["opening_kills"] / total_duels,
    )
    df["opening_duel_win_rate"] = np.round(df["opening_duel_win_rate"], 4)
    logger.info(
        "Computed 'opening_duel_win_rate' — mean=%.3f, std=%.3f",
        df["opening_duel_win_rate"].mean(), df["opening_duel_win_rate"].std(),
    )
    return df


def compute_consistency_score(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Consistency Score.

    Uses a rating-deviation proxy normalised to [0, 1]:

    ``raw = 1 / (1 + |rating − 1.0|)``

    Then min-max normalised across all players.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        DataFrame with an added ``consistency_score`` column.
    """
    df = df.copy()
    raw = 1.0 / (1.0 + np.abs(df["rating"] - 1.0))
    raw_min, raw_max = raw.min(), raw.max()

    if raw_max - raw_min == 0:
        df["consistency_score"] = 1.0
    else:
        df["consistency_score"] = np.round((raw - raw_min) / (raw_max - raw_min), 4)

    logger.info(
        "Computed 'consistency_score' — mean=%.3f, std=%.3f",
        df["consistency_score"].mean(), df["consistency_score"].std(),
    )
    return df


def _min_max_normalize(series: pd.Series) -> pd.Series:
    """Min-max normalise a Series to [0, 1]."""
    s_min, s_max = series.min(), series.max()
    if s_max - s_min == 0:
        return pd.Series(0.5, index=series.index)
    return (series - s_min) / (s_max - s_min)


def compute_performance_index(df: pd.DataFrame) -> pd.DataFrame:
    """Compute composite Performance Index.

    ``PI = 0.35·rating_norm + 0.25·kdr_norm + 0.20·impact_norm + 0.20·hs_norm``

    Each component is min-max normalised before weighting.

    Parameters
    ----------
    df : pd.DataFrame
        Must already contain ``kdr`` and ``impact_score`` columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with an added ``performance_index`` column.
    """
    df = df.copy()
    rating_norm = _min_max_normalize(df["rating"])
    kdr_norm = _min_max_normalize(df["kdr"])
    impact_norm = _min_max_normalize(df["impact_score"])
    hs_norm = _min_max_normalize(df["headshot_percentage"])

    df["performance_index"] = np.round(
        0.35 * rating_norm
        + 0.25 * kdr_norm
        + 0.20 * impact_norm
        + 0.20 * hs_norm,
        4,
    )
    logger.info(
        "Computed 'performance_index' — mean=%.3f, std=%.3f",
        df["performance_index"].mean(), df["performance_index"].std(),
    )
    return df


# ============================================================================
# Target variable
# ============================================================================

def create_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """Create the classification target from ``performance_index``.

    Quantile-based bucketing:

    * Bottom 30 % → **Low** (0)
    * Middle 40 % → **Medium** (1)
    * Top 30 %    → **High** (2)

    Adds two columns: ``performance_category`` (str) and
    ``performance_encoded`` (int).

    Parameters
    ----------
    df : pd.DataFrame
        Must already contain ``performance_index``.

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    low_thresh = df["performance_index"].quantile(LOW_QUANTILE)
    high_thresh = df["performance_index"].quantile(HIGH_QUANTILE)

    conditions = [
        df["performance_index"] <= low_thresh,
        df["performance_index"] >= high_thresh,
    ]
    choices_encoded = [0, 2]  # Low=0, High=2
    df[TARGET_COLUMN_ENCODED] = np.select(conditions, choices_encoded, default=1)

    # String labels
    df[TARGET_COLUMN] = df[TARGET_COLUMN_ENCODED].map(PERFORMANCE_CATEGORIES)

    counts = df[TARGET_COLUMN].value_counts()
    logger.info(
        "Target variable created — thresholds: low=%.4f, high=%.4f", low_thresh, high_thresh,
    )
    for cat in ["Low", "Medium", "High"]:
        logger.info("  %s: %d (%.1f%%)", cat, counts.get(cat, 0), 100 * counts.get(cat, 0) / len(df))

    return df


# ============================================================================
# Unified pipeline
# ============================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature-engineering pipeline.

    Computes all engineered features and creates the target variable.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed (cleaned) DataFrame.

    Returns
    -------
    pd.DataFrame
        Enriched DataFrame with new feature and target columns.
    """
    logger.info("=" * 60)
    logger.info("Starting feature engineering (%d rows).", len(df))
    logger.info("=" * 60)

    df = compute_kdr(df)
    df = compute_impact_score(df)
    df = compute_survival_rate(df)
    df = compute_assist_contribution(df)
    df = compute_opening_duel_win_rate(df)
    df = compute_consistency_score(df)
    df = compute_performance_index(df)
    df = create_target_variable(df)

    logger.info("Feature engineering complete — final shape: %s", df.shape)
    return df


# ============================================================================
# CLI entry-point
# ============================================================================

if __name__ == "__main__":
    setup_logging()

    logger.info("Loading processed data from %s", PROCESSED_DATA_PATH)
    df = load_dataframe(PROCESSED_DATA_PATH)

    df = engineer_features(df)

    save_dataframe(df, ENRICHED_DATA_PATH)
    print(f"\nEnriched dataset: {len(df)} rows × {df.shape[1]} columns")
    print(f"Saved to {ENRICHED_DATA_PATH}")
    print("\nNew columns:")
    new_cols = [
        "kdr", "impact_score", "survival_rate", "assist_contribution",
        "opening_duel_win_rate", "consistency_score", "performance_index",
        TARGET_COLUMN, TARGET_COLUMN_ENCODED,
    ]
    print(df[new_cols].describe().round(4).to_string())
