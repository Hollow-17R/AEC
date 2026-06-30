"""
Model Training for E-Sports Player Performance Classification.

Trains three classifiers – Logistic Regression, Random Forest, and XGBoost –
on the processed CS:GO player statistics and persists all artefacts.
"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.utils import (
    get_logger,
    save_model,
    load_dataframe,
    print_section_header,
    PROCESSED_DATA_PATH,
    MODELS_DIR,
    SCALER_PATH,
    FEATURE_COLUMNS_PATH,
    MODEL_FEATURES,
    TARGET_COLUMN_ENCODED,
    MODEL_NAMES,
    RANDOM_STATE,
    TEST_SIZE,
)

logger = get_logger(__name__)


# ============================================================================
# Data Preparation
# ============================================================================


def prepare_data(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split the processed DataFrame into train/test arrays.

    Steps
    -----
    1. Select ``MODEL_FEATURES`` columns as X and ``TARGET_COLUMN_ENCODED`` as y.
    2. Perform a stratified train/test split.
    3. Fit a ``StandardScaler`` on the training split and persist it.
    4. Save the feature column list for inference-time validation.

    Parameters
    ----------
    df : pd.DataFrame
        Processed player-stats DataFrame.

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
        Scaled feature arrays and corresponding target arrays.
    """
    features = [f for f in MODEL_FEATURES if f in df.columns]
    X = df[features].values
    y = df[TARGET_COLUMN_ENCODED].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )

    # Fit and persist scaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    joblib.dump(scaler, SCALER_PATH)
    logger.info("Scaler saved to %s", SCALER_PATH)

    joblib.dump(features, FEATURE_COLUMNS_PATH)
    logger.info("Feature columns saved to %s", FEATURE_COLUMNS_PATH)

    logger.info(
        "Data split → train=%d, test=%d, features=%d",
        X_train.shape[0], X_test.shape[0], X_train.shape[1],
    )
    return X_train, X_test, y_train, y_test


# ============================================================================
# Individual Model Trainers
# ============================================================================


def train_logistic_regression(
    X_train: np.ndarray, y_train: np.ndarray
) -> Pipeline:
    """Train a Logistic Regression pipeline with internal scaling.

    Parameters
    ----------
    X_train : np.ndarray
        Training features (already scaled externally, but pipeline adds its own).
    y_train : np.ndarray
        Training labels.

    Returns
    -------
    Pipeline
        Fitted sklearn Pipeline.
    """
    logger.info("Training Logistic Regression …")
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            multi_class="multinomial",
            max_iter=1000,
            random_state=RANDOM_STATE,
            solver="lbfgs",
        )),
    ])
    pipeline.fit(X_train, y_train)
    logger.info("Logistic Regression training complete.")
    return pipeline


def train_random_forest(
    X_train: np.ndarray, y_train: np.ndarray
) -> RandomForestClassifier:
    """Train a Random Forest classifier.

    Parameters
    ----------
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training labels.

    Returns
    -------
    RandomForestClassifier
        Fitted model.
    """
    logger.info("Training Random Forest …")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    logger.info("Random Forest training complete.")
    return model


def train_xgboost(
    X_train: np.ndarray, y_train: np.ndarray
) -> XGBClassifier:
    """Train an XGBoost classifier for multi-class classification.

    Parameters
    ----------
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training labels.

    Returns
    -------
    XGBClassifier
        Fitted model.
    """
    logger.info("Training XGBoost …")
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        use_label_encoder=False,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    logger.info("XGBoost training complete.")
    return model


# ============================================================================
# Train All
# ============================================================================


def train_all_models(
    X_train: np.ndarray, y_train: np.ndarray
) -> Dict[str, object]:
    """Train all three classifiers and return them in a dictionary.

    Parameters
    ----------
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training labels.

    Returns
    -------
    dict
        ``{model_name: fitted_model}`` for each of the three classifiers.
    """
    print_section_header("MODEL TRAINING")
    models: Dict[str, object] = {}

    models[MODEL_NAMES[0]] = train_logistic_regression(X_train, y_train)
    models[MODEL_NAMES[1]] = train_random_forest(X_train, y_train)
    models[MODEL_NAMES[2]] = train_xgboost(X_train, y_train)

    # Persist every trained model
    for name, model in models.items():
        filename = name.lower().replace(" ", "_") + ".pkl"
        save_model(model, MODELS_DIR / filename)

    logger.info("All %d models trained and saved.", len(models))
    return models


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    from src.utils import setup_logging

    setup_logging()
    logger.info("Loading processed data …")
    df = load_dataframe(PROCESSED_DATA_PATH)

    X_train, X_test, y_train, y_test = prepare_data(df)
    models = train_all_models(X_train, y_train)

    # Quick accuracy check
    for name, model in models.items():
        acc = model.score(X_test, y_test)
        logger.info("%s → Test accuracy: %.4f", name, acc)
