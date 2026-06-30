"""
E-Sports Player Performance Classification - Main Pipeline Runner
================================================================
CLI entry point to run the entire ML pipeline end-to-end or individual steps.

Usage:
    python main.py --all          # Run complete pipeline
    python main.py --generate     # Generate synthetic dataset
    python main.py --preprocess   # Preprocess raw data
    python main.py --engineer     # Feature engineering
    python main.py --eda          # Generate EDA visualizations
    python main.py --train        # Train all models
    python main.py --tune         # Hyperparameter tuning
    python main.py --evaluate     # Evaluate models
    python main.py --explain      # Generate SHAP explanations
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import (
    setup_logging,
    get_logger,
    print_section_header,
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    BEST_MODEL_PATH,
)


def step_generate():
    """Step 1: Generate synthetic CS:GO dataset."""
    print_section_header("STEP 1: DATA GENERATION")
    from src.data_generator import generate_dataset

    df = generate_dataset()
    print(f"  [OK] Generated {len(df)} player records")
    print(f"  [OK] Saved to {RAW_DATA_PATH}")
    return df


def step_preprocess():
    """Step 2: Preprocess raw data."""
    print_section_header("STEP 2: DATA PREPROCESSING")
    from src.data_preprocessing import preprocess_pipeline
    from src.utils import load_dataframe

    df = load_dataframe(RAW_DATA_PATH)
    df_clean = preprocess_pipeline(df)
    print(f"  [OK] Preprocessed {len(df_clean)} records")
    return df_clean


def step_engineer():
    """Step 3: Feature engineering."""
    print_section_header("STEP 3: FEATURE ENGINEERING")
    from src.feature_engineering import engineer_features
    from src.utils import load_dataframe, save_dataframe, PROCESSED_DATA_PATH

    df = load_dataframe(RAW_DATA_PATH)

    # Run preprocessing first if needed
    from src.data_preprocessing import preprocess_pipeline
    df = preprocess_pipeline(df)

    # Engineer features
    df_engineered = engineer_features(df)
    save_dataframe(df_engineered, PROCESSED_DATA_PATH)
    print(f"  [OK] Engineered {len(df_engineered.columns)} features")
    print(f"  [OK] Target distribution:")
    if "performance_category" in df_engineered.columns:
        counts = df_engineered["performance_category"].value_counts()
        for cat, count in counts.items():
            pct = count / len(df_engineered) * 100
            print(f"      {cat}: {count} ({pct:.1f}%)")
    return df_engineered


def step_eda():
    """Step 4: Exploratory Data Analysis."""
    print_section_header("STEP 4: EXPLORATORY DATA ANALYSIS")
    from src.eda import run_eda
    from src.utils import load_dataframe

    df = load_dataframe(PROCESSED_DATA_PATH)
    run_eda(df)
    print("  [OK] All EDA visualizations generated")


def step_train():
    """Step 5: Train all models."""
    print_section_header("STEP 5: MODEL TRAINING")
    from src.model_training import prepare_data, train_all_models
    from src.utils import load_dataframe, save_model, MODELS_DIR

    df = load_dataframe(PROCESSED_DATA_PATH)
    X_train, X_test, y_train, y_test = prepare_data(df)

    models = train_all_models(X_train, y_train)

    # Save train/test split for evaluation
    import joblib
    joblib.dump((X_train, X_test, y_train, y_test), MODELS_DIR / "train_test_split.pkl")

    for name, model in models.items():
        safe_name = name.lower().replace(" ", "_")
        save_model(model, MODELS_DIR / f"{safe_name}.pkl")
        print(f"  [OK] {name} trained and saved")

    return models, X_train, X_test, y_train, y_test


def step_tune():
    """Step 6: Hyperparameter tuning."""
    print_section_header("STEP 6: HYPERPARAMETER TUNING")
    from src.hyperparameter_tuning import tune_all_models, save_best_model
    from src.utils import load_dataframe, MODELS_DIR

    import joblib
    X_train, X_test, y_train, y_test = joblib.load(MODELS_DIR / "train_test_split.pkl")

    tuned_models = tune_all_models(X_train, y_train)
    save_best_model(tuned_models, X_test, y_test)
    print("  [OK] Hyperparameter tuning complete")
    print(f"  [OK] Best model saved to {BEST_MODEL_PATH}")

    return tuned_models


def step_evaluate():
    """Step 7: Evaluate all models."""
    print_section_header("STEP 7: MODEL EVALUATION")
    from src.model_evaluation import evaluate_all_models
    from src.utils import load_model, MODELS_DIR

    import joblib
    X_train, X_test, y_train, y_test = joblib.load(MODELS_DIR / "train_test_split.pkl")

    import numpy as np
    X_full = np.vstack([X_train, X_test])
    y_full = np.concatenate([y_train, y_test])

    # Load models
    models = {}
    model_files = {
        "Logistic Regression": "logistic_regression.pkl",
        "Random Forest": "random_forest.pkl",
        "XGBoost": "xgboost.pkl",
    }

    for name, fname in model_files.items():
        model_path = MODELS_DIR / fname
        if model_path.exists():
            models[name] = load_model(model_path)

    # Also load tuned models if available
    tuned_files = {
        "Logistic Regression (Tuned)": "logistic_regression_tuned.pkl",
        "Random Forest (Tuned)": "random_forest_tuned.pkl",
        "XGBoost (Tuned)": "xgboost_tuned.pkl",
    }
    for name, fname in tuned_files.items():
        model_path = MODELS_DIR / fname
        if model_path.exists():
            models[name] = load_model(model_path)

    if not models:
        print("  [ERROR] No trained models found. Run --train first.")
        return

    results_df = evaluate_all_models(models, X_test, y_test, X_full, y_full)
    print("\n  Model Comparison:")
    print(results_df.to_string(index=False))
    print("  [OK] Evaluation complete")


def step_explain():
    """Step 8: Generate explainability visualizations."""
    print_section_header("STEP 8: EXPLAINABILITY ANALYSIS")
    from src.explainability import run_explainability
    from src.utils import load_model, MODELS_DIR, BEST_MODEL_PATH

    import joblib
    X_train, X_test, y_train, y_test = joblib.load(MODELS_DIR / "train_test_split.pkl")

    # Load best model
    if BEST_MODEL_PATH.exists():
        best_model = load_model(BEST_MODEL_PATH)
    else:
        # Fallback to Random Forest
        rf_path = MODELS_DIR / "random_forest.pkl"
        if rf_path.exists():
            best_model = load_model(rf_path)
        else:
            print("  [ERROR] No model found. Run --train first.")
            return

    # Load feature columns
    feature_cols_path = MODELS_DIR / "feature_columns.pkl"
    if feature_cols_path.exists():
        feature_columns = joblib.load(feature_cols_path)
    else:
        from src.utils import MODEL_FEATURES
        feature_columns = MODEL_FEATURES

    run_explainability(best_model, X_test, feature_columns)
    print("  [OK] Explainability analysis complete")


def run_all():
    """Run the complete pipeline end-to-end."""
    start_time = time.time()

    print("\n" + "=" * 70)
    print("  E-SPORTS PLAYER PERFORMANCE CLASSIFICATION")
    print("  Full Pipeline Execution")
    print("=" * 70)

    step_generate()
    step_engineer()  # includes preprocessing
    step_eda()
    step_train()
    step_tune()
    step_evaluate()
    step_explain()

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"  [OK] PIPELINE COMPLETE - Total time: {elapsed:.1f}s")
    print("=" * 70)
    print(f"\n  To launch the dashboard:")
    print(f"    streamlit run app/app.py")
    print()


def main():
    """Parse arguments and run requested pipeline steps."""
    setup_logging()

    parser = argparse.ArgumentParser(
        description="E-Sports Player Performance Classification Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --all          Run complete pipeline
  python main.py --generate     Generate synthetic dataset only
  python main.py --train --tune Train and tune models
        """,
    )

    parser.add_argument("--generate", action="store_true", help="Generate synthetic dataset")
    parser.add_argument("--preprocess", action="store_true", help="Preprocess raw data")
    parser.add_argument("--engineer", action="store_true", help="Run feature engineering")
    parser.add_argument("--eda", action="store_true", help="Generate EDA visualizations")
    parser.add_argument("--train", action="store_true", help="Train all models")
    parser.add_argument("--tune", action="store_true", help="Hyperparameter tuning")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate models")
    parser.add_argument("--explain", action="store_true", help="Generate SHAP explanations")
    parser.add_argument("--all", action="store_true", help="Run complete pipeline")

    args = parser.parse_args()

    # If no arguments provided, show help
    if not any(vars(args).values()):
        parser.print_help()
        sys.exit(0)

    if args.all:
        run_all()
        return

    if args.generate:
        step_generate()
    if args.preprocess:
        step_preprocess()
    if args.engineer:
        step_engineer()
    if args.eda:
        step_eda()
    if args.train:
        step_train()
    if args.tune:
        step_tune()
    if args.evaluate:
        step_evaluate()
    if args.explain:
        step_explain()


if __name__ == "__main__":
    main()
