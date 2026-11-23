# Delivery Delay Prediction | Prédiction des retards de livraison

> **EN:** Machine-learning project to classify a delivery as on time or delayed using logistics features available before dispatch.  
> **FR :** Projet de machine learning qui classe une livraison comme effectuée à temps ou en retard à partir de variables logistiques disponibles avant le départ.

## 🇬🇧 English

### Overview
The repository contains a preprocessing module, a Random Forest training pipeline, a prediction helper and a Streamlit interface. The model is trained with scikit-learn using a preprocessing pipeline, one-hot encoding for categorical features, median imputation for numerical features and a GridSearchCV hyperparameter search.

### Prediction inputs
- Distance in kilometres
- Delivery zone
- Day of the week
- Departure time
- Number of parcels
- Weather / traffic condition
- Planned delivery time

### Important target-leakage correction
The dataset includes `delai_reel` (actual delivery time), which is only known after a delivery has taken place. The training pipeline now explicitly excludes this column, as well as the target `statut`, from model inputs. The Streamlit form no longer sends a placeholder `delai_reel` value to the predictor.

### Files
| File | Purpose |
|---|---|
| `streamlit_app.py` | Streamlit prediction interface |
| `train_model.py` | Model training and hyperparameter search |
| `preprocessing.py` | Numerical and categorical preprocessing |
| `predict.py` | Loads the saved pipeline and predicts one delivery |
| `mini_projet_logistique.csv` | Project dataset |
| `requirements.txt` | Python dependencies |

### Setup and run
```bash
git clone https://github.com/Slymanidiallo/Pr-diction-des-d-lais-de-livraisons.git
cd Pr-diction-des-d-lais-de-livraisons
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python train_model.py
streamlit run streamlit_app.py
```

The training script reads the CSV from the repository root and saves the fitted pipeline to `models/model.pkl`. The application expects this model file to exist.

### Evaluation caveat
GridSearchCV uses stratified three-fold cross-validation and reports its best cross-validation F1 score. This is not a separate final test-set result. A future evaluation should reserve a held-out test set and assess class-wise precision/recall, F1, calibration and robustness on new delivery records.

## 🇫🇷 Français

### Présentation
Le dépôt contient un module de prétraitement, un pipeline d'entraînement Random Forest, une fonction de prédiction et une interface Streamlit. Le modèle est entraîné avec scikit-learn à l'aide d'un pipeline qui comprend le prétraitement, le one-hot encoding des variables catégorielles, l'imputation médiane des variables numériques et une recherche d'hyperparamètres par GridSearchCV.

### Variables utilisées pour la prédiction
- Distance en kilomètres
- Zone de livraison
- Jour de la semaine
- Heure de départ
- Nombre de colis
- Météo / état du trafic
- Délai prévu

### Correction importante de fuite de la cible
Le jeu de données contient la variable `delai_reel` (durée réelle de livraison), qui n'est connue qu'après la livraison. Le pipeline d'entraînement exclut désormais explicitement cette colonne ainsi que la cible `statut` des variables explicatives. Le formulaire Streamlit n'envoie plus de valeur provisoire `delai_reel` au prédicteur.

### Fichiers
| Fichier | Rôle |
|---|---|
| `streamlit_app.py` | Interface de prédiction Streamlit |
| `train_model.py` | Entraînement et recherche d'hyperparamètres |
| `preprocessing.py` | Prétraitement numérique et catégoriel |
| `predict.py` | Chargement du pipeline sauvegardé et prédiction |
| `mini_projet_logistique.csv` | Jeu de données |
| `requirements.txt` | Dépendances Python |

### Installation et lancement
```bash
git clone https://github.com/Slymanidiallo/Pr-diction-des-d-lais-de-livraisons.git
cd Pr-diction-des-d-lais-de-livraisons
python -m venv .venv
# Windows : .venv\Scripts\activate
# macOS/Linux : source .venv/bin/activate
pip install -r requirements.txt

python train_model.py
streamlit run streamlit_app.py
```

Le script d'entraînement lit le CSV depuis la racine du dépôt et sauvegarde le pipeline dans `models/model.pkl`. L'application attend la présence de ce fichier.

### Précaution sur l'évaluation
GridSearchCV utilise une validation croisée stratifiée en trois plis et affiche le meilleur score F1 de validation croisée. Ce résultat ne constitue pas une performance mesurée sur un jeu de test indépendant. Une prochaine évaluation devra réserver un jeu de test final et examiner la précision, le rappel, le F1-score par classe, la calibration et la robustesse sur de nouvelles livraisons.
