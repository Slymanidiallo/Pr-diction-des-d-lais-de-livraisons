from pathlib import Path

import joblib
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "model.pkl"


def predict_single(data: dict):
    """Predict delivery status from features available before the delivery."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. Run 'python train_model.py' first."
        )

    model = joblib.load(MODEL_PATH)
    row = pd.DataFrame([data])
    expected_columns = list(model.feature_names_in_)
    missing = sorted(set(expected_columns).difference(row.columns))
    if missing:
        raise ValueError(f"Missing prediction features: {missing}")

    row = row.reindex(columns=expected_columns)
    prediction = int(model.predict(row)[0])
    probabilities = model.predict_proba(row)[0]
    classes = list(model.classes_)
    delay_index = classes.index(1)
    return prediction, float(probabilities[delay_index])
