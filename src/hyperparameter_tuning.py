"""
Hyperparameter Tuning for E-Sports Player Performance Classification.

Uses GridSearchCV / RandomizedSearchCV with stratified 5-fold cross-validation
to find optimal parameters for Logistic Regression, Random Forest, and XGBoost.
The overall best model is saved to ``BEST_MODEL_PATH``.
"""

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier

from src.utils import (
    get_logger,
    save_model,
    load_dataframe,
    print_section_header,
    PROCESSED_DATA_PATH,
    MODELS_DIR,
    BEST_MODEL_PATH,
    MODEL_FEATURES,
    TARGET_COLUMN_ENCODED,
    MODEL_NAMES,
    RANDOM_STATE,
    TEST_SIZE,
)

logger = get_logger(__name__)

# Shared CV splitter
CV_FOLDS = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


# ============================================================================
# Logistic Regression Tuning (GridSearchCV)
# ============================================================================


def tune_logistic_regression(
    X_train: np.ndarray, y_train: np.ndarray
) -> GridSearchCV:
    """Tune Logistic Regression via exhaustive grid search.

    Parameters
    ----------
    X_train : np.ndarray
        Training features (scaled).
    y_train : np.ndarray
        Training labels.

    Returns
    -------
    GridSearchCV
        Fitted grid search object. Access ``.best_estimator_`` for the best
        pipeline and ``.best_params_`` for the winning parameters.
    """
    logger.info("Tuning Logistic Regression (GridSearchCV) …")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            multi_class="multinomial",
            max_iter=1000,
            random_state=RANDOM_STATE,
        )),
    ])

    param_grid = {
        "clf__C": [0.01, 0.1, 1, 10, 100],
        "clf__penalty": ["l2"],
        "clf__solver": ["lbfgs", "newton-cg"],
    }

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=CV_FOLDS,
        scoring="f1_weighted",
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    grid.fit(X_train, y_train)

    logger.info(
        "Logistic Regression best params: %s  |  best CV F1: %.4f",
        grid.best_params_, grid.best_score_,
    )
    return grid


# ============================================================================
# Random Forest Tuning (RandomizedSearchCV)
# ============================================================================


def tune_random_forest(
    X_train: np.ndarray, y_train: np.ndarray
) -> RandomizedSearchCV:
    """Tune Random Forest via randomised search.

    Parameters
    ----------
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training labels.

    Returns
    -------
    RandomizedSearchCV
        Fitted search object.
    """
    logger.info("Tuning Random Forest (RandomizedSearchCV, n_iter=50) …")

    model = RandomForestClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    param_distributions = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [5, 10, 15, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=50,
        cv=CV_FOLDS,
        scoring="f1_weighted",
        n_jobs=-1,
        verbose=1,
        random_state=RANDOM_STATE,
        refit=True,
    )
    search.fit(X_train, y_train)

    logger.info(
        "Random Forest best params: %s  |  best CV F1: %.4f",
        search.best_params_, search.best_score_,
    )
    return search


# ============================================================================
# XGBoost Tuning (RandomizedSearchCV)
# ============================================================================


def tune_xgboost(
    X_train: np.ndarray, y_train: np.ndarray
) -> RandomizedSearchCV:
    """Tune XGBoost via randomised search.

    Parameters
    ----------
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training labels.

    Returns
    -------
    RandomizedSearchCV
        Fitted search object.
    """
    logger.info("Tuning XGBoost (RandomizedSearchCV, n_iter=50) …")

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    param_distributions = {
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 5, 7, 9],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5],
    }

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=50,
        cv=CV_FOLDS,
        scoring="f1_weighted",
        n_jobs=-1,
        verbose=1,
        random_state=RANDOM_STATE,
        refit=True,
    )
    search.fit(X_train, y_train)

    logger.info(
        "XGBoost best params: %s  |  best CV F1: %.4f",
        search.best_params_, search.best_score_,
    )
    return search


# ============================================================================
# Tune All Models
# ============================================================================


def tune_all_models(
    X_train: np.ndarray, y_train: np.ndarray
) -> Dict[str, Any]:
    """Run hyperparameter tuning for all three classifiers.

    Parameters
    ----------
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training labels.

    Returns
    -------
    dict
        ``{model_name: best_estimator}`` for each classifier.
    """
    print_section_header("HYPERPARAMETER TUNING")
    tuned_models: Dict[str, Any] = {}

    # Logistic Regression
    lr_search = tune_logistic_regression(X_train, y_train)
    tuned_models[MODEL_NAMES[0]] = lr_search.best_estimator_
    print(f"\n  {MODEL_NAMES[0]}")
    print(f"    Best params : {lr_search.best_params_}")
    print(f"    Best CV F1  : {lr_search.best_score_:.4f}")

    # Random Forest
    rf_search = tune_random_forest(X_train, y_train)
    tuned_models[MODEL_NAMES[1]] = rf_search.best_estimator_
    print(f"\n  {MODEL_NAMES[1]}")
    print(f"    Best params : {rf_search.best_params_}")
    print(f"    Best CV F1  : {rf_search.best_score_:.4f}")

    # XGBoost
    xgb_search = tune_xgboost(X_train, y_train)
    tuned_models[MODEL_NAMES[2]] = xgb_search.best_estimator_
    print(f"\n  {MODEL_NAMES[2]}")
    print(f"    Best params : {xgb_search.best_params_}")
    print(f"    Best CV F1  : {xgb_search.best_score_:.4f}")

    # Save tuned models
    for name, model in tuned_models.items():
        filename = name.lower().replace(" ", "_") + "_tuned.pkl"
        save_model(model, MODELS_DIR / filename)

    logger.info("All tuned models saved to %s", MODELS_DIR)
    return tuned_models


# ============================================================================
# Save Best Model
# ============================================================================


def save_best_model(
    tuned_models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[str, Any]:
    """Evaluate all tuned models on the test set and persist the best one.

    Parameters
    ----------
    tuned_models : dict
        ``{model_name: fitted_model}``.
    X_test : np.ndarray
        Test features.
    y_test : np.ndarray
        True labels.

    Returns
    -------
    tuple[str, estimator]
        Name and model object of the best performer.
    """
    print_section_header("SELECTING BEST MODEL")
    best_name: str = ""
    best_f1: float = -1.0
    best_model: Any = None

    for name, model in tuned_models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        print(f"  {name:25s}  →  Accuracy: {acc:.4f}  |  F1: {f1:.4f}")
        logger.info("%s → Accuracy=%.4f, F1=%.4f", name, acc, f1)

        if f1 > best_f1:
            best_f1 = f1
            best_name = name
            best_model = model

    save_model(best_model, BEST_MODEL_PATH)
    print(f"\n  ★ Best model: {best_name} (F1={best_f1:.4f})")
    print(f"    Saved to: {BEST_MODEL_PATH}")
    logger.info("Best model '%s' saved to %s", best_name, BEST_MODEL_PATH)

    return best_name, best_model


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    from src.utils import setup_logging

    setup_logging()

    logger.info("Loading processed data …")
    df = load_dataframe(PROCESSED_DATA_PATH)

    features = [f for f in MODEL_FEATURES if f in df.columns]
    X = df[features].values
    y = df[TARGET_COLUMN_ENCODED].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Tune
    tuned_models = tune_all_models(X_train, y_train)

    # Select and save the champion
    best_name, best_model = save_best_model(tuned_models, X_test, y_test)
