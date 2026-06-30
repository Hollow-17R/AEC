"""
Data Preprocessing for E-Sports Player Performance Classification.

Provides a full cleaning pipeline: missing-value imputation, duplicate
removal, outlier capping (IQR), data validation, and a single
``preprocess_pipeline()`` function that chains every step.
"""

from typing import List

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

from src.utils import (
    ORIGINAL_FEATURES,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
    get_logger,
    load_dataframe,
    save_dataframe,
    setup_logging,
)

logger = get_logger(__name__)

# Columns that must never be negative
_NON_NEGATIVE_COLS: List[str] = [
    "maps_played", "rounds_played", "total_kills", "total_deaths",
    "total_assists", "headshot_percentage", "kills_per_round",
    "deaths_per_round", "assists_per_round", "damage_per_round",
    "rating", "opening_kills", "opening_deaths", "clutch_wins",
]


# ============================================================================
# Pipeline steps
# ============================================================================

def load_raw_data() -> pd.DataFrame:
    """Load raw CS:GO player data from ``RAW_DATA_PATH``.

    Returns
    -------
    pd.DataFrame
        Raw dataset as read from CSV.
    """
    logger.info("Loading raw data from %s", RAW_DATA_PATH)
    df = load_dataframe(RAW_DATA_PATH)
    logger.info("Raw data shape: %s", df.shape)
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values.

    * **Numeric columns** – KNN imputation (k=5).
    * **Categorical columns** – mode (most frequent value).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (may contain NaN values).

    Returns
    -------
    pd.DataFrame
        DataFrame with no missing values.
    """
    df = df.copy()
    missing_before = int(df.isna().sum().sum())
    logger.info("Missing values before imputation: %d", missing_before)

    if missing_before == 0:
        logger.info("No missing values detected — skipping imputation.")
        return df

    # --- Numeric imputation (KNN) ---
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        n_missing_num = int(df[numeric_cols].isna().sum().sum())
        if n_missing_num > 0:
            logger.info(
                "Applying KNN imputation to %d numeric column(s) (%d NaNs).",
                len(numeric_cols), n_missing_num,
            )
            imputer = KNNImputer(n_neighbors=5, weights="uniform")
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    # --- Categorical imputation (mode) ---
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        n_miss = int(df[col].isna().sum())
        if n_miss > 0:
            mode_val = df[col].mode().iloc[0]
            df[col].fillna(mode_val, inplace=True)
            logger.info("Column '%s': filled %d NaNs with mode '%s'.", col, n_miss, mode_val)

    missing_after = int(df.isna().sum().sum())
    logger.info("Missing values after imputation: %d", missing_after)
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Deduplicated DataFrame.
    """
    df = df.copy()
    n_before = len(df)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    n_removed = n_before - len(df)
    logger.info("Duplicates removed: %d (rows %d → %d).", n_removed, n_before, len(df))
    return df


def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Cap outliers using the IQR method.

    For every numeric column in ``ORIGINAL_FEATURES``, values beyond
    ``[Q1 − 1.5·IQR, Q3 + 1.5·IQR]`` are clipped to the bounds.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        DataFrame with outliers capped.
    """
    df = df.copy()
    numeric_cols = [c for c in ORIGINAL_FEATURES if c in df.columns]
    total_capped = 0

    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        n_low = int((df[col] < lower).sum())
        n_high = int((df[col] > upper).sum())
        n_capped = n_low + n_high

        if n_capped > 0:
            df[col] = df[col].clip(lower=lower, upper=upper)
            logger.info(
                "Column '%s': capped %d outlier(s) [%d low, %d high] "
                "to [%.4f, %.4f].",
                col, n_capped, n_low, n_high, lower, upper,
            )
            total_capped += n_capped

    logger.info("Total outlier values capped: %d across %d column(s).", total_capped, len(numeric_cols))
    return df


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate data types, value ranges, and integrity.

    Checks performed:
    1. Expected numeric columns are actually numeric.
    2. No negative values in columns that should be ≥ 0.
    3. ``headshot_percentage`` is within [0, 100].
    4. ``rating`` is positive.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        The (potentially corrected) DataFrame.

    Raises
    ------
    ValueError
        If a fatal validation issue is detected.
    """
    df = df.copy()
    issues: List[str] = []

    # 1. Numeric type check
    for col in _NON_NEGATIVE_COLS:
        if col in df.columns and not np.issubdtype(df[col].dtype, np.number):
            issues.append(f"Column '{col}' is not numeric (dtype={df[col].dtype}).")

    # 2. Non-negative check
    for col in _NON_NEGATIVE_COLS:
        if col in df.columns:
            n_neg = int((df[col] < 0).sum())
            if n_neg > 0:
                logger.warning(
                    "Column '%s' has %d negative value(s) — clipping to 0.", col, n_neg,
                )
                df[col] = df[col].clip(lower=0)

    # 3. Headshot percentage range
    if "headshot_percentage" in df.columns:
        out_of_range = int(((df["headshot_percentage"] < 0) | (df["headshot_percentage"] > 100)).sum())
        if out_of_range > 0:
            logger.warning(
                "'headshot_percentage' has %d value(s) outside [0, 100] — clipping.", out_of_range,
            )
            df["headshot_percentage"] = df["headshot_percentage"].clip(0, 100)

    # 4. Rating positive
    if "rating" in df.columns:
        n_non_pos = int((df["rating"] <= 0).sum())
        if n_non_pos > 0:
            logger.warning("'rating' has %d non-positive value(s) — setting to 0.01.", n_non_pos)
            df.loc[df["rating"] <= 0, "rating"] = 0.01

    if issues:
        msg = "Data validation issues:\n" + "\n".join(f"  • {i}" for i in issues)
        logger.error(msg)
        raise ValueError(msg)

    logger.info("Data validation passed ✓ (%d rows, %d columns).", *df.shape)
    return df


# ============================================================================
# Unified pipeline
# ============================================================================

def preprocess_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full preprocessing pipeline in order.

    Steps executed:
    1. Handle missing values (KNN / mode imputation).
    2. Remove duplicate rows.
    3. Cap outliers (IQR method).
    4. Validate data integrity.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input DataFrame.

    Returns
    -------
    pd.DataFrame
        Cleaned & validated DataFrame.
    """
    logger.info("=" * 60)
    logger.info("Starting preprocessing pipeline (%d rows).", len(df))
    logger.info("=" * 60)

    # Step 1
    logger.info("— Step 1/4: Handling missing values")
    df = handle_missing_values(df)

    # Step 2
    logger.info("— Step 2/4: Removing duplicates")
    df = remove_duplicates(df)

    # Step 3
    logger.info("— Step 3/4: Handling outliers (IQR capping)")
    df = handle_outliers(df)

    # Step 4
    logger.info("— Step 4/4: Validating data")
    df = validate_data(df)

    logger.info("Preprocessing complete — final shape: %s", df.shape)
    return df


# ============================================================================
# CLI entry-point
# ============================================================================

if __name__ == "__main__":
    setup_logging()
    raw_df = load_raw_data()
    clean_df = preprocess_pipeline(raw_df)
    save_dataframe(clean_df, PROCESSED_DATA_PATH)
    print(f"\nPreprocessed {len(clean_df)} rows -> {PROCESSED_DATA_PATH}")
    print(clean_df.describe().round(2).to_string())
