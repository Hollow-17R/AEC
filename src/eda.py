"""
Exploratory Data Analysis (EDA) for E-Sports Player Performance Classification.

Generates publication-quality dark-themed visualizations to understand
player statistics distributions, correlations, and category separability.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif

from src.utils import (
    get_logger,
    set_plot_style,
    save_figure,
    load_dataframe,
    print_section_header,
    PROCESSED_DATA_PATH,
    FIGURES_DIR,
    MODEL_FEATURES,
    TARGET_COLUMN,
    TARGET_COLUMN_ENCODED,
    PERFORMANCE_COLORS,
    CATEGORY_LABELS,
)

logger = get_logger(__name__)

# Accent palette for non-category colouring
CYAN = "#00d2ff"
MAGENTA = "#ff00e5"
GREEN = "#00ff87"
ACCENT_PALETTE = [CYAN, MAGENTA, GREEN, "#f0c000", "#ff4757", "#7c4dff"]


# ============================================================================
# 1. Feature Distributions
# ============================================================================


def plot_distributions(df: pd.DataFrame) -> None:
    """Plot histograms with KDE overlays for the top 12 numeric features.

    Renders a 4×3 grid of distribution plots for the first 12 features
    listed in MODEL_FEATURES, saved as ``feature_distributions.png``.

    Parameters
    ----------
    df : pd.DataFrame
        Processed player-stats DataFrame.
    """
    set_plot_style()
    features = [f for f in MODEL_FEATURES if f in df.columns][:12]
    fig, axes = plt.subplots(4, 3, figsize=(20, 18))
    fig.patch.set_facecolor("#0e1117")

    for idx, (ax, feat) in enumerate(zip(axes.ravel(), features)):
        color = ACCENT_PALETTE[idx % len(ACCENT_PALETTE)]
        sns.histplot(df[feat], kde=True, ax=ax, color=color, edgecolor="none", alpha=0.7)
        ax.set_title(feat.replace("_", " ").title(), fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")

    # Hide any unused subplot slots
    for ax in axes.ravel()[len(features):]:
        ax.set_visible(False)

    fig.suptitle("Feature Distributions", fontsize=20, fontweight="bold", color=CYAN, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, "feature_distributions.png")
    plt.close(fig)
    logger.info("Saved feature_distributions.png")


# ============================================================================
# 2. Correlation Heatmap
# ============================================================================


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Plot an annotated correlation matrix heatmap for all model features.

    Parameters
    ----------
    df : pd.DataFrame
        Processed player-stats DataFrame.
    """
    set_plot_style()
    numeric_cols = [c for c in MODEL_FEATURES if c in df.columns]
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(18, 15))
    fig.patch.set_facecolor("#0e1117")

    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(250, 15, s=90, l=40, n=12, as_cmap=True)

    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        linecolor="#30304a",
        cbar_kws={"shrink": 0.8, "label": "Pearson r"},
        annot_kws={"size": 8},
        ax=ax,
    )

    ax.set_title("Feature Correlation Matrix", fontsize=20, fontweight="bold", color=CYAN, pad=20)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    fig.tight_layout()
    save_figure(fig, "correlation_heatmap.png")
    plt.close(fig)
    logger.info("Saved correlation_heatmap.png")


# ============================================================================
# 3. Box Plots by Category
# ============================================================================


def plot_boxplots_by_category(df: pd.DataFrame) -> None:
    """Box plots of key features split by performance category (2×3 grid).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``performance_category`` column.
    """
    set_plot_style()
    key_features = ["rating", "kdr", "impact_score", "damage_per_round", "kills_per_round", "headshot_percentage"]
    key_features = [f for f in key_features if f in df.columns][:6]

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.patch.set_facecolor("#0e1117")

    palette = PERFORMANCE_COLORS

    for ax, feat in zip(axes.ravel(), key_features):
        sns.boxplot(
            data=df,
            x=TARGET_COLUMN,
            y=feat,
            order=CATEGORY_LABELS,
            palette=palette,
            ax=ax,
            flierprops={"marker": "o", "markerfacecolor": "#ffffff40", "markersize": 3},
            boxprops={"alpha": 0.8},
        )
        ax.set_title(feat.replace("_", " ").title(), fontsize=13, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")

    for ax in axes.ravel()[len(key_features):]:
        ax.set_visible(False)

    fig.suptitle("Feature Distributions by Performance Category", fontsize=20, fontweight="bold", color=CYAN, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, "boxplots_by_category.png")
    plt.close(fig)
    logger.info("Saved boxplots_by_category.png")


# ============================================================================
# 4. Scatter Analysis
# ============================================================================


def plot_scatter_analysis(df: pd.DataFrame) -> None:
    """Two scatter plots coloured by performance category.

    Left : KDR vs Rating.
    Right: Impact Score vs Performance Index.

    Parameters
    ----------
    df : pd.DataFrame
        Processed player-stats DataFrame.
    """
    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.patch.set_facecolor("#0e1117")

    palette = PERFORMANCE_COLORS

    # Panel 1 – KDR vs Rating
    if "kdr" in df.columns and "rating" in df.columns:
        sns.scatterplot(
            data=df, x="kdr", y="rating",
            hue=TARGET_COLUMN, hue_order=CATEGORY_LABELS,
            palette=palette, alpha=0.6, s=30, edgecolor="none", ax=axes[0],
        )
        axes[0].set_title("KDR vs Rating", fontsize=14, fontweight="bold")
        axes[0].set_xlabel("Kill/Death Ratio")
        axes[0].set_ylabel("HLTV Rating")

    # Panel 2 – Impact Score vs Performance Index
    if "impact_score" in df.columns and "performance_index" in df.columns:
        sns.scatterplot(
            data=df, x="impact_score", y="performance_index",
            hue=TARGET_COLUMN, hue_order=CATEGORY_LABELS,
            palette=palette, alpha=0.6, s=30, edgecolor="none", ax=axes[1],
        )
        axes[1].set_title("Impact Score vs Performance Index", fontsize=14, fontweight="bold")
        axes[1].set_xlabel("Impact Score")
        axes[1].set_ylabel("Performance Index")

    fig.suptitle("Scatter Analysis by Performance Category", fontsize=20, fontweight="bold", color=CYAN, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, "scatter_analysis.png")
    plt.close(fig)
    logger.info("Saved scatter_analysis.png")


# ============================================================================
# 5. Violin Plots
# ============================================================================


def plot_violin_plots(df: pd.DataFrame) -> None:
    """Violin plots for rating, kdr, and impact_score by category.

    Parameters
    ----------
    df : pd.DataFrame
        Processed player-stats DataFrame.
    """
    set_plot_style()
    violin_features = ["rating", "kdr", "impact_score"]
    violin_features = [f for f in violin_features if f in df.columns]

    fig, axes = plt.subplots(1, len(violin_features), figsize=(20, 7))
    fig.patch.set_facecolor("#0e1117")

    if len(violin_features) == 1:
        axes = [axes]

    palette = PERFORMANCE_COLORS

    for ax, feat in zip(axes, violin_features):
        sns.violinplot(
            data=df, x=TARGET_COLUMN, y=feat,
            order=CATEGORY_LABELS, palette=palette,
            inner="box", linewidth=1.2, ax=ax,
        )
        ax.set_title(feat.replace("_", " ").title(), fontsize=14, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")

    fig.suptitle("Violin Plots by Performance Category", fontsize=20, fontweight="bold", color=CYAN, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, "violin_plots.png")
    plt.close(fig)
    logger.info("Saved violin_plots.png")


# ============================================================================
# 6. Target Distribution
# ============================================================================


def plot_target_distribution(df: pd.DataFrame) -> None:
    """Bar chart of High / Medium / Low class counts with percentage labels.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``performance_category`` column.
    """
    set_plot_style()
    counts = df[TARGET_COLUMN].value_counts().reindex(CATEGORY_LABELS)
    total = counts.sum()
    percentages = (counts / total * 100).round(1)

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#0e1117")

    bars = ax.bar(
        counts.index, counts.values,
        color=[PERFORMANCE_COLORS[c] for c in counts.index],
        edgecolor="none", width=0.55, alpha=0.9,
    )

    for bar, pct, cnt in zip(bars, percentages.values, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.01,
            f"{cnt}\n({pct}%)",
            ha="center", va="bottom", fontsize=13, fontweight="bold", color="#e0e0e0",
        )

    ax.set_title("Target Variable Distribution", fontsize=20, fontweight="bold", color=CYAN, pad=15)
    ax.set_ylabel("Number of Players", fontsize=13)
    ax.set_xlabel("Performance Category", fontsize=13)
    ax.set_ylim(0, counts.max() * 1.18)
    fig.tight_layout()
    save_figure(fig, "target_distribution.png")
    plt.close(fig)
    logger.info("Saved target_distribution.png")


# ============================================================================
# 7. Country Distribution
# ============================================================================


def plot_country_distribution(df: pd.DataFrame) -> None:
    """Horizontal bar chart of the top 15 countries by player count.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``country`` column.
    """
    set_plot_style()

    if "country" not in df.columns:
        logger.warning("'country' column not found – skipping country distribution plot.")
        return

    top15 = df["country"].value_counts().head(15).sort_values()

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor("#0e1117")

    colors = sns.color_palette("viridis", n_colors=len(top15))
    ax.barh(top15.index, top15.values, color=colors, edgecolor="none", height=0.6)

    for i, (val, name) in enumerate(zip(top15.values, top15.index)):
        ax.text(val + top15.max() * 0.01, i, str(val), va="center", fontsize=11, color="#e0e0e0")

    ax.set_title("Top 15 Countries by Player Count", fontsize=20, fontweight="bold", color=CYAN, pad=15)
    ax.set_xlabel("Number of Players", fontsize=13)
    ax.set_ylabel("")
    fig.tight_layout()
    save_figure(fig, "country_distribution.png")
    plt.close(fig)
    logger.info("Saved country_distribution.png")


# ============================================================================
# 8. Pair Plot
# ============================================================================


def plot_pairplot(df: pd.DataFrame) -> None:
    """Pair plot of the top 5 model features coloured by performance category.

    Parameters
    ----------
    df : pd.DataFrame
        Processed player-stats DataFrame.
    """
    set_plot_style()
    top_features = [f for f in MODEL_FEATURES if f in df.columns][:5]
    plot_df = df[top_features + [TARGET_COLUMN]].dropna()

    palette = PERFORMANCE_COLORS
    g = sns.pairplot(
        plot_df,
        hue=TARGET_COLUMN,
        hue_order=CATEGORY_LABELS,
        palette=palette,
        diag_kind="kde",
        plot_kws={"alpha": 0.5, "s": 15, "edgecolor": "none"},
        diag_kws={"alpha": 0.6, "linewidth": 1.5},
        corner=False,
    )
    g.figure.patch.set_facecolor("#0e1117")
    g.figure.suptitle("Pair Plot – Top 5 Features", fontsize=18, fontweight="bold", color=CYAN, y=1.01)
    save_figure(g.figure, "pairplot.png")
    plt.close(g.figure)
    logger.info("Saved pairplot.png")


# ============================================================================
# 9. Feature Importance Preview (Mutual Information)
# ============================================================================


def plot_feature_importance_preview(df: pd.DataFrame) -> None:
    """Bar chart of mutual-information scores between features and the target.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``MODEL_FEATURES`` columns and ``TARGET_COLUMN_ENCODED``.
    """
    set_plot_style()
    features = [f for f in MODEL_FEATURES if f in df.columns]
    X = df[features].copy()
    y = df[TARGET_COLUMN_ENCODED].copy()

    # Drop rows with NaN for MI calculation
    valid = X.notna().all(axis=1) & y.notna()
    X, y = X.loc[valid], y.loc[valid]

    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_series = pd.Series(mi_scores, index=features).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(12, 9))
    fig.patch.set_facecolor("#0e1117")

    colors = [CYAN if v >= mi_series.quantile(0.75) else MAGENTA if v >= mi_series.median() else "#f0c000" for v in mi_series.values]
    ax.barh(mi_series.index, mi_series.values, color=colors, edgecolor="none", height=0.6)

    for i, val in enumerate(mi_series.values):
        ax.text(val + mi_series.max() * 0.01, i, f"{val:.3f}", va="center", fontsize=10, color="#e0e0e0")

    ax.set_title("Feature Importance (Mutual Information)", fontsize=20, fontweight="bold", color=CYAN, pad=15)
    ax.set_xlabel("MI Score", fontsize=13)
    ax.set_ylabel("")
    fig.tight_layout()
    save_figure(fig, "feature_importance_preview.png")
    plt.close(fig)
    logger.info("Saved feature_importance_preview.png")


# ============================================================================
# Run All EDA
# ============================================================================


def run_eda(df: pd.DataFrame) -> None:
    """Execute the complete EDA pipeline and print summary statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Fully processed player-stats DataFrame.
    """
    print_section_header("EXPLORATORY DATA ANALYSIS")
    logger.info("Starting EDA pipeline …")

    # ---- Summary Statistics ------------------------------------------------
    print_section_header("Dataset Overview")
    print(f"Shape         : {df.shape}")
    print(f"Columns       : {list(df.columns)}")
    print(f"Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

    numeric_cols = [c for c in MODEL_FEATURES if c in df.columns]
    print_section_header("Descriptive Statistics")
    print(df[numeric_cols].describe().round(3).to_string())

    if TARGET_COLUMN in df.columns:
        print_section_header("Target Distribution")
        print(df[TARGET_COLUMN].value_counts().to_string())

    # ---- Generate All Plots ------------------------------------------------
    plot_distributions(df)
    plot_correlation_heatmap(df)
    plot_boxplots_by_category(df)
    plot_scatter_analysis(df)
    plot_violin_plots(df)
    plot_target_distribution(df)
    plot_country_distribution(df)
    plot_pairplot(df)
    plot_feature_importance_preview(df)

    logger.info("EDA pipeline completed – all figures saved to %s", FIGURES_DIR)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    from src.utils import setup_logging

    setup_logging()
    logger.info("Loading processed data for EDA …")
    df = load_dataframe(PROCESSED_DATA_PATH)
    run_eda(df)
