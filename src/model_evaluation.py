"""
Model Evaluation for E-Sports Player Performance Classification.

Computes classification metrics, generates confusion matrices, ROC curves,
learning curves, and a side-by-side model comparison chart.
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import label_binarize

from src.utils import (
    get_logger,
    set_plot_style,
    save_figure,
    load_dataframe,
    load_model,
    print_section_header,
    PROCESSED_DATA_PATH,
    MODELS_DIR,
    FIGURES_DIR,
    MODEL_FEATURES,
    TARGET_COLUMN_ENCODED,
    MODEL_NAMES,
    CATEGORY_LABELS,
    PERFORMANCE_COLORS,
    RANDOM_STATE,
    TEST_SIZE,
)

logger = get_logger(__name__)

# Accent colours
CYAN = "#00d2ff"
MAGENTA = "#ff00e5"
GREEN = "#00ff87"
CLASS_COLORS = ["#ff4757", "#f0c000", "#00ff87"]  # Low, Medium, High


# ============================================================================
# Single-Model Evaluation
# ============================================================================


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
) -> Dict[str, Any]:
    """Compute classification metrics for a single model.

    Parameters
    ----------
    model : estimator
        Trained sklearn/XGBoost model.
    X_test : np.ndarray
        Test features.
    y_test : np.ndarray
        True labels.
    model_name : str
        Human-readable model identifier.

    Returns
    -------
    dict
        Keys: accuracy, precision, recall, f1, classification_report.
    """
    y_pred = model.predict(X_test)

    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "classification_report": classification_report(
            y_test, y_pred, target_names=CATEGORY_LABELS, zero_division=0,
        ),
    }

    logger.info(
        "%s → Accuracy=%.4f  Precision=%.4f  Recall=%.4f  F1=%.4f",
        model_name, results["accuracy"], results["precision"],
        results["recall"], results["f1"],
    )
    print(f"\n{'─'*50}")
    print(f"  {model_name} – Classification Report")
    print(f"{'─'*50}")
    print(results["classification_report"])

    return results


# ============================================================================
# Confusion Matrix
# ============================================================================


def plot_confusion_matrix(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
) -> None:
    """Plot and save an annotated confusion-matrix heatmap.

    Parameters
    ----------
    model : estimator
        Trained model with a ``predict`` method.
    X_test, y_test : np.ndarray
        Test data.
    model_name : str
        Used for the title and filename.
    """
    set_plot_style()
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor("#0e1117")

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="magma",
        xticklabels=CATEGORY_LABELS,
        yticklabels=CATEGORY_LABELS,
        linewidths=1,
        linecolor="#30304a",
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 14, "fontweight": "bold"},
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=13)
    ax.set_ylabel("Actual", fontsize=13)
    ax.set_title(
        f"{model_name} – Confusion Matrix",
        fontsize=16, fontweight="bold", color=CYAN, pad=15,
    )
    fig.tight_layout()

    fname = f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
    save_figure(fig, fname)
    plt.close(fig)
    logger.info("Saved %s", fname)


# ============================================================================
# ROC Curves (One-vs-Rest)
# ============================================================================


def plot_roc_curves(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
) -> None:
    """Plot One-vs-Rest ROC curves for all three classes.

    Falls back gracefully if the model does not support ``predict_proba``.

    Parameters
    ----------
    model : estimator
        Trained model.
    X_test, y_test : np.ndarray
        Test data.
    model_name : str
        Used for the title and filename.
    """
    set_plot_style()

    if not hasattr(model, "predict_proba"):
        logger.warning("%s does not support predict_proba – skipping ROC curves.", model_name)
        return

    y_prob = model.predict_proba(X_test)
    classes = sorted(np.unique(y_test))
    y_bin = label_binarize(y_test, classes=classes)

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#0e1117")

    for idx, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, idx], y_prob[:, idx])
        roc_auc = auc(fpr, tpr)
        label_name = CATEGORY_LABELS[idx] if idx < len(CATEGORY_LABELS) else str(cls)
        ax.plot(
            fpr, tpr,
            color=CLASS_COLORS[idx % len(CLASS_COLORS)],
            linewidth=2.5,
            label=f"{label_name} (AUC = {roc_auc:.3f})",
        )

    ax.plot([0, 1], [0, 1], linestyle="--", color="#555555", linewidth=1)
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title(
        f"{model_name} – ROC Curves (One-vs-Rest)",
        fontsize=16, fontweight="bold", color=CYAN, pad=15,
    )
    ax.legend(fontsize=12, loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    fig.tight_layout()

    fname = f"{model_name.lower().replace(' ', '_')}_roc_curves.png"
    save_figure(fig, fname)
    plt.close(fig)
    logger.info("Saved %s", fname)


# ============================================================================
# Model Comparison Chart
# ============================================================================


def plot_model_comparison(results_dict: Dict[str, Dict[str, Any]]) -> None:
    """Grouped bar chart comparing all models across evaluation metrics.

    Parameters
    ----------
    results_dict : dict
        ``{model_name: {metric_name: value, …}, …}``
    """
    set_plot_style()
    metrics = ["accuracy", "precision", "recall", "f1"]
    model_names = list(results_dict.keys())

    data = {m: [results_dict[name][m] for name in model_names] for m in metrics}
    x = np.arange(len(model_names))
    width = 0.18

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor("#0e1117")

    colors = [CYAN, MAGENTA, GREEN, "#f0c000"]
    for i, metric in enumerate(metrics):
        bars = ax.bar(
            x + i * width, data[metric], width,
            label=metric.capitalize(), color=colors[i], alpha=0.85, edgecolor="none",
        )
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=9, color="#e0e0e0",
            )

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names, fontsize=12)
    ax.set_ylabel("Score", fontsize=13)
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=12, loc="upper left")
    ax.set_title(
        "Model Comparison – All Metrics",
        fontsize=18, fontweight="bold", color=CYAN, pad=15,
    )
    fig.tight_layout()
    save_figure(fig, "model_comparison.png")
    plt.close(fig)
    logger.info("Saved model_comparison.png")


# ============================================================================
# Learning Curves
# ============================================================================


def plot_learning_curves(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
) -> None:
    """Plot training vs validation accuracy learning curves.

    Parameters
    ----------
    model : estimator
        Trained (or untrained) sklearn-compatible model.
    X, y : np.ndarray
        Full dataset (train + test).
    model_name : str
        Used for the title and filename.
    """
    set_plot_style()

    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("#0e1117")

    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color=CYAN)
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color=MAGENTA)
    ax.plot(train_sizes, train_mean, "o-", color=CYAN, linewidth=2, label="Training Accuracy")
    ax.plot(train_sizes, val_mean, "o-", color=MAGENTA, linewidth=2, label="Validation Accuracy")

    ax.set_xlabel("Training Set Size", fontsize=13)
    ax.set_ylabel("Accuracy", fontsize=13)
    ax.set_title(
        f"{model_name} – Learning Curves",
        fontsize=16, fontweight="bold", color=CYAN, pad=15,
    )
    ax.legend(fontsize=12, loc="lower right")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()

    fname = f"{model_name.lower().replace(' ', '_')}_learning_curves.png"
    save_figure(fig, fname)
    plt.close(fig)
    logger.info("Saved %s", fname)


# ============================================================================
# Evaluate All Models
# ============================================================================


def evaluate_all_models(
    models_dict: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_full: np.ndarray,
    y_full: np.ndarray,
) -> pd.DataFrame:
    """Evaluate every model, generate plots, and return a comparison DataFrame.

    Parameters
    ----------
    models_dict : dict
        ``{model_name: fitted_model}``.
    X_test, y_test : np.ndarray
        Test split arrays.
    X_full, y_full : np.ndarray
        Full dataset for learning curves.

    Returns
    -------
    pd.DataFrame
        Comparison table with one row per model and metric columns.
    """
    print_section_header("MODEL EVALUATION")
    all_results: Dict[str, Dict[str, Any]] = {}

    for name, model in models_dict.items():
        logger.info("Evaluating %s …", name)
        results = evaluate_model(model, X_test, y_test, name)
        all_results[name] = results

        plot_confusion_matrix(model, X_test, y_test, name)
        plot_roc_curves(model, X_test, y_test, name)
        plot_learning_curves(model, X_full, y_full, name)

    # Cross-model comparison chart
    plot_model_comparison(all_results)

    # Build comparison DataFrame
    comparison_rows = []
    for name, res in all_results.items():
        comparison_rows.append({
            "Model": name,
            "Accuracy": round(res["accuracy"], 4),
            "Precision": round(res["precision"], 4),
            "Recall": round(res["recall"], 4),
            "F1 Score": round(res["f1"], 4),
        })
    comparison_df = pd.DataFrame(comparison_rows)

    print_section_header("MODEL COMPARISON SUMMARY")
    print(comparison_df.to_string(index=False))

    logger.info("Model evaluation complete.")
    return comparison_df


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
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
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Combine for learning curves
    X_full = np.vstack([X_train, X_test])
    y_full = np.concatenate([y_train, y_test])

    # Load trained models
    models: Dict[str, Any] = {}
    for name in MODEL_NAMES:
        fname = name.lower().replace(" ", "_") + ".pkl"
        try:
            models[name] = load_model(MODELS_DIR / fname)
        except FileNotFoundError:
            logger.warning("Model file %s not found – skipping.", fname)

    if models:
        comparison_df = evaluate_all_models(models, X_test, y_test, X_full, y_full)
    else:
        logger.error("No trained models found. Run model_training.py first.")
