import os
import pandas as pd
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from preprocessing import build_preprocessor

# Chemins sûrs
PROJECT_DIR = os.path.expanduser("~/Bureau/project_logistique ")  # espace final inclus
DATA_PATH = os.path.join(PROJECT_DIR, "Data", "mini_projet_logistique.csv")
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")

# Charger les données
df = pd.read_csv(DATA_PATH)

# Prétraitement et modèle
X = df.drop(columns=["statut"])
y = df["statut"].map({"dans les temps": 0, "retard": 1})
categorical_features = [c for c in X.columns if X[c].dtype == "object"]
numeric_features = [c for c in X.columns if c not in categorical_features]
preprocessor = build_preprocessor(categorical_features, numeric_features)
clf = RandomForestClassifier(random_state=42)
pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])

# Grid search
param_grid = {"classifier__n_estimators": [100, 200], "classifier__max_depth": [None, 10, 20]}
search = GridSearchCV(pipeline, param_grid, cv=3, scoring="f1", n_jobs=-1)
search.fit(X, y)

print("Meilleurs paramètres:", search.best_params_)

# Sauvegarder le modèle
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(search.best_estimator_, MODEL_PATH)
print(f"✅ Modèle sauvegardé dans {MODEL_PATH}")
