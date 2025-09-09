import pandas as pd
import joblib

def predict_single(data: dict):
    model = joblib.load("models/model.pkl")  # chemin relatif depuis racine
    df = pd.DataFrame([data])
    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0][1]  # probabilité de retard
    return pred, proba

