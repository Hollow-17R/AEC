"""
Explainability Module for E-Sports Player Performance Classification.
====================================================================
Generates SHAP-based explanations and feature importance visualizations
to understand which factors most influence player performance predictions.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings

from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import (
    get_logger,
    save_figure,
    set_plot_style,
    FIGURES_DIR,
    MODELS_DIR,
    BEST_MODEL_PATH,
    CATEGORY_LABELS,
    PERFORMANCE_COLORS,
    MODEL_FEATURES,
    load_model,
)

logger = get_logger(__name__)


def get_shap_explainer(model, X_sample):
    """
    Create appropriate SHAP explainer based on model type.
    
    Parameters
    ----------
    model : trained model
        The model to explain.
    X_sample : np.ndarray
        Background data sample for the explainer.
    
    Returns
    -------
    tuple : (explainer, shap_values)
    """
    import shap

    model_type = type(model).__name__

    try:
        if model_type == "Pipeline":
            # For sklearn Pipeline (e.g., Logistic Regression with scaler)
            # Use the full pipeline as a function
            explainer = shap.KernelExplainer(
                model.predict_proba,
                shap.sample(X_sample, min(100, len(X_sample))),
            )
            shap_values = explainer.shap_values(X_sample[:min(200, len(X_sample))])
        elif model_type in ["RandomForestClassifier", "GradientBoostingClassifier"]:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample[:min(500, len(X_sample))])
        elif model_type == "XGBClassifier":
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample[:min(500, len(X_sample))])
        else:
            # Generic fallback
            explainer = shap.KernelExplainer(
                model.predict_proba,
                shap.sample(X_sample, min(50, len(X_sample))),
            )
            shap_values = explainer.shap_values(X_sample[:min(100, len(X_sample))])
    except Exception as e:
        logger.warning(f"SHAP explainer failed for {model_type}: {e}")
        logger.info("Falling back to permutation-based importance")
        return None, None

    return explainer, shap_values


def plot_shap_summary(shap_values, X_data, feature_names, class_idx=2):
    """
    Generate SHAP summary (beeswarm) plot for a specific class.
    
    Parameters
    ----------
    shap_values : array-like
        SHAP values from explainer.
    X_data : np.ndarray
        Feature data used for SHAP computation.
    feature_names : list
        Feature column names.
    class_idx : int
        Class index to plot (default 2 = High performance).
    """
    import shap

    set_plot_style()
    fig, ax = plt.subplots(figsize=(12, 8))

    # Get SHAP values for the target class
    if isinstance(shap_values, list):
        sv = shap_values[class_idx]
    else:
        sv = shap_values

    # Create DataFrame for plotting
    if len(X_data) > len(sv):
        X_plot = X_data[:len(sv)]
    else:
        X_plot = X_data

    shap.summary_plot(
        sv,
        X_plot,
        feature_names=feature_names,
        show=False,
        plot_size=(12, 8),
        color_bar_label="Feature Value",
    )

    plt.title(
        f"SHAP Feature Impact — {CATEGORY_LABELS[class_idx]} Performance",
        fontsize=16,
        fontweight="bold",
        color="#e0e0e0",
        pad=20,
    )
    plt.tight_layout()
    save_figure(plt.gcf(), "shap_summary_plot.png")
    plt.close()
    logger.info("SHAP summary plot saved")


def plot_shap_bar(shap_values, feature_names):
    """
    Generate SHAP bar plot showing mean absolute SHAP values.
    
    Parameters
    ----------
    shap_values : array-like
        SHAP values from explainer.
    feature_names : list
        Feature column names.
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(12, 8))

    # Compute mean absolute SHAP values across all classes
    if isinstance(shap_values, list):
        # Multi-class: average across classes
        mean_shap = np.mean(
            [np.abs(sv).mean(axis=0) for sv in shap_values], axis=0
        )
    else:
        mean_shap = np.abs(shap_values).mean(axis=0)

    # Sort features by importance
    sorted_idx = np.argsort(mean_shap)
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_values = mean_shap[sorted_idx]

    # Create gradient colors
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(sorted_features)))

    bars = ax.barh(range(len(sorted_features)), sorted_values, color=colors, edgecolor="none")

    ax.set_yticks(range(len(sorted_features)))
    ax.set_yticklabels(sorted_features, fontsize=11)
    ax.set_xlabel("Mean |SHAP Value|", fontsize=13, fontweight="bold")
    ax.set_title(
        "Feature Importance — SHAP Analysis",
        fontsize=16,
        fontweight="bold",
        color="#e0e0e0",
        pad=20,
    )

    # Add value labels
    for bar, val in zip(bars, sorted_values):
        ax.text(
            bar.get_width() + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            fontsize=9,
            color="#b0b0b0",
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    save_figure(fig, "shap_bar_plot.png")
    plt.close()
    logger.info("SHAP bar plot saved")


def plot_builtin_feature_importance(model, feature_names, model_name="Model"):
    """
    Plot built-in feature importance for tree-based models.
    
    Parameters
    ----------
    model : trained model
        Must have `feature_importances_` attribute.
    feature_names : list
        Feature column names.
    model_name : str
        Name of the model for the title.
    """
    set_plot_style()

    if not hasattr(model, "feature_importances_"):
        logger.warning(f"{model_name} does not have built-in feature importances")
        return

    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_importances = importances[sorted_idx]

    fig, ax = plt.subplots(figsize=(12, 8))

    # Gradient color scheme
    colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(sorted_features)))

    bars = ax.barh(range(len(sorted_features)), sorted_importances, color=colors, edgecolor="none")

    ax.set_yticks(range(len(sorted_features)))
    ax.set_yticklabels(sorted_features, fontsize=11)
    ax.set_xlabel("Feature Importance", fontsize=13, fontweight="bold")
    ax.set_title(
        f"Built-in Feature Importance — {model_name}",
        fontsize=16,
        fontweight="bold",
        color="#e0e0e0",
        pad=20,
    )

    # Add percentage labels
    total = sorted_importances.sum()
    for bar, val in zip(bars, sorted_importances):
        pct = val / total * 100
        ax.text(
            bar.get_width() + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}%",
            va="center",
            fontsize=9,
            color="#b0b0b0",
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    safe_name = model_name.lower().replace(" ", "_")
    save_figure(fig, f"{safe_name}_feature_importance.png")
    plt.close()
    logger.info(f"Built-in feature importance plot saved for {model_name}")


def plot_partial_dependence(model, X_data, feature_names, top_n=3):
    """
    Generate Partial Dependence Plots for the top N most important features.
    
    Parameters
    ----------
    model : trained model
        The model to analyze.
    X_data : np.ndarray
        Feature data.
    feature_names : list
        Feature column names.
    top_n : int
        Number of top features to plot.
    """
    from sklearn.inspection import PartialDependenceDisplay

    set_plot_style()

    # Get feature importances to find top features
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        # Use permutation importance as fallback
        from sklearn.inspection import permutation_importance
        try:
            # Need y for permutation importance, use a simple proxy
            perm_result = permutation_importance(model, X_data[:200], model.predict(X_data[:200]), n_repeats=5, random_state=42)
            importances = perm_result.importances_mean
        except Exception:
            importances = np.ones(len(feature_names))

    top_indices = np.argsort(importances)[-top_n:][::-1]
    top_feature_names = [feature_names[i] for i in top_indices]

    fig, axes = plt.subplots(1, top_n, figsize=(6 * top_n, 5))
    if top_n == 1:
        axes = [axes]

    fig.patch.set_facecolor("#0e1117")

    try:
        display = PartialDependenceDisplay.from_estimator(
            model,
            X_data[:500],
            features=top_indices.tolist(),
            feature_names=feature_names,
            ax=axes,
            grid_resolution=50,
            kind="average",
        )

        for ax in axes:
            ax.set_facecolor("#1a1a2e")
            ax.tick_params(colors="#b0b0b0")
            ax.xaxis.label.set_color("#e0e0e0")
            ax.yaxis.label.set_color("#e0e0e0")
            ax.title.set_color("#e0e0e0")

        fig.suptitle(
            "Partial Dependence — Top Features",
            fontsize=16,
            fontweight="bold",
            color="#e0e0e0",
            y=1.02,
        )

        plt.tight_layout()
        save_figure(fig, "partial_dependence_plots.png")
        plt.close()
        logger.info("Partial dependence plots saved")
    except Exception as e:
        logger.warning(f"Partial dependence plot failed: {e}")
        plt.close()


def plot_feature_correlation_with_target(df, feature_names, target_col="performance_encoded"):
    """
    Plot correlation of each feature with the target variable.
    
    Parameters
    ----------
    df : pd.DataFrame
        Processed DataFrame with features and target.
    feature_names : list
        Feature column names.
    target_col : str
        Target column name.
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(12, 8))

    available = [f for f in feature_names if f in df.columns]
    if target_col not in df.columns:
        logger.warning(f"Target column '{target_col}' not found")
        return

    correlations = df[available].corrwith(df[target_col]).sort_values()

    colors = ["#ff4757" if v < 0 else "#00ff87" for v in correlations.values]

    bars = ax.barh(range(len(correlations)), correlations.values, color=colors, edgecolor="none")

    ax.set_yticks(range(len(correlations)))
    ax.set_yticklabels(correlations.index, fontsize=11)
    ax.set_xlabel("Correlation with Performance", fontsize=13, fontweight="bold")
    ax.set_title(
        "Feature Correlation with Target Variable",
        fontsize=16,
        fontweight="bold",
        color="#e0e0e0",
        pad=20,
    )

    ax.axvline(x=0, color="#ffffff", linewidth=0.8, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    save_figure(fig, "feature_target_correlation.png")
    plt.close()
    logger.info("Feature-target correlation plot saved")


def run_explainability(model, X_test, feature_columns, df=None):
    """
    Run complete explainability analysis.
    
    Parameters
    ----------
    model : trained model
        The model to explain.
    X_test : np.ndarray
        Test feature data.
    feature_columns : list
        Feature column names.
    df : pd.DataFrame, optional
        Full processed DataFrame for correlation analysis.
    """
    logger.info("Starting explainability analysis...")

    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # 1. Built-in feature importance (for tree models)
    model_to_explain = model
    model_name = type(model).__name__

    # Handle Pipeline objects
    if model_name == "Pipeline":
        # Try to get the final estimator
        try:
            model_to_explain = model.named_steps.get("classifier", model[-1])
            model_name = type(model_to_explain).__name__
        except Exception:
            pass

    plot_builtin_feature_importance(model_to_explain, feature_columns, model_name)

    # 2. SHAP analysis
    logger.info("Computing SHAP values (this may take a moment)...")
    try:
        explainer, shap_values = get_shap_explainer(model, X_test)

        if shap_values is not None:
            # SHAP summary plot
            plot_shap_summary(shap_values, X_test, feature_columns, class_idx=2)

            # SHAP bar plot
            plot_shap_bar(shap_values, feature_columns)
            logger.info("SHAP analysis complete")
        else:
            logger.warning("SHAP analysis skipped — explainer returned None")
    except Exception as e:
        logger.warning(f"SHAP analysis failed: {e}")
        logger.info("Continuing with other explainability methods...")

    # 3. Partial Dependence Plots
    try:
        plot_partial_dependence(model_to_explain, X_test, feature_columns, top_n=3)
    except Exception as e:
        logger.warning(f"Partial dependence plots failed: {e}")

    # 4. Feature-target correlation (if DataFrame provided)
    if df is not None:
        try:
            plot_feature_correlation_with_target(df, feature_columns)
        except Exception as e:
            logger.warning(f"Feature-target correlation plot failed: {e}")

    warnings.filterwarnings("default")
    logger.info("Explainability analysis complete")


# ============================================================================
# Main execution
# ============================================================================
if __name__ == "__main__":
    import joblib
    from src.utils import setup_logging, load_dataframe, PROCESSED_DATA_PATH

    setup_logging()
    print_section_header = lambda t: print(f"\n{'='*60}\n  {t}\n{'='*60}")
    print_section_header("EXPLAINABILITY ANALYSIS")

    # Load best model
    if BEST_MODEL_PATH.exists():
        model = load_model(BEST_MODEL_PATH)
    else:
        # Try Random Forest
        rf_path = MODELS_DIR / "random_forest.pkl"
        if rf_path.exists():
            model = load_model(rf_path)
        else:
            print("No model found. Run model training first.")
            sys.exit(1)

    # Load test data
    split_path = MODELS_DIR / "train_test_split.pkl"
    if split_path.exists():
        X_train, X_test, y_train, y_test = joblib.load(split_path)
    else:
        print("No train/test split found. Run model training first.")
        sys.exit(1)

    # Load feature columns
    feature_cols_path = MODELS_DIR / "feature_columns.pkl"
    if feature_cols_path.exists():
        feature_columns = joblib.load(feature_cols_path)
    else:
        feature_columns = MODEL_FEATURES

    # Load processed data for correlation analysis
    df = None
    if PROCESSED_DATA_PATH.exists():
        df = load_dataframe(PROCESSED_DATA_PATH)

    run_explainability(model, X_test, feature_columns, df)
    print("\n✓ Explainability analysis complete. Check reports/figures/")
