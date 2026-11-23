from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from preprocessing import build_preprocessor


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "mini_projet_logistique.csv"
MODEL_DIR = PROJECT_DIR / "models"
MODEL_PATH = MODEL_DIR / "model.pkl"

TARGET_COLUMN = "statut"
# delai_reel is only known after delivery and must never be used as a predictor.
UNAVAILABLE_AT_PREDICTION = ["delai_reel"]


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. Place mini_projet_logistique.csv "
            "in the project root."
        )

    df = pd.read_csv(DATA_PATH)
    required = {TARGET_COLUMN, "delai_reel"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required dataset columns: {sorted(missing)}")

    y = df[TARGET_COLUMN].astype(str).str.strip().str.lower().map({
        "dans les temps": 0,
        "retard": 1,
    })
    if y.isna().any():
        unknown = sorted(df.loc[y.isna(), TARGET_COLUMN].astype(str).unique())
        raise ValueError(f"Unknown target labels in '{TARGET_COLUMN}': {unknown}")

    X = df.drop(columns=[TARGET_COLUMN, *UNAVAILABLE_AT_PREDICTION])
    if X.empty:
        raise ValueError("No predictor columns remain after excluding target/leakage.")

    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_features = [c for c in X.columns if c not in categorical_features]
    preprocessor = build_preprocessor(categorical_features, numeric_features)

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42)),
    ])
    param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [None, 10, 20],
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    search = GridSearchCV(
        pipeline, param_grid, cv=cv, scoring="f1", n_jobs=-1, error_score="raise"
    )
    search.fit(X, y)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(search.best_estimator_, MODEL_PATH)
    print("Best parameters:", search.best_params_)
    print(f"Model saved to: {MODEL_PATH}")
    print("Training F1 (cross-validation best score):", round(search.best_score_, 4))


if __name__ == "__main__":
    main()
