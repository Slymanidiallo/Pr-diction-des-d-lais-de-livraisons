# Delivery Delay Prediction | Prédiction des retards de livraison

> **EN:** Machine-learning project to classify a delivery as on time or delayed from logistics-related input features.  
> **FR :** Projet de machine learning visant à estimer si une livraison sera effectuée à temps ou avec retard à partir de variables logistiques.

## 🇬🇧 English

### Overview
The repository includes a preprocessing module, a model-training script, a prediction helper and a Streamlit interface. The training script builds a scikit-learn pipeline with preprocessing and a **Random Forest classifier**, tuning selected hyperparameters with **GridSearchCV**.

### Application inputs
The Streamlit form asks for:
- Distance in kilometres
- Delivery zone
- Day of the week
- Departure time
- Number of parcels
- Weather / traffic condition
- Planned delivery time

The interface displays an on-time or delayed status and a delay probability returned by the prediction helper.

### Project files
| File | Purpose |
|---|---|
| `streamlit_app.py` | User interface |
| `train_model.py` | Pipeline training and hyperparameter search |
| `preprocessing.py` | Feature preprocessing |
| `predict.py` | Single-record prediction helper |
| `mini_projet_logistique.csv` | Dataset |
| `requirements.txt` | Python dependencies |

### Setup
```bash
git clone https://github.com/Slymanidiallo/Pr-diction-des-d-lais-de-livraisons.git
cd Pr-diction-des-d-lais-de-livraisons
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Before training, update the dataset/model paths in `train_model.py`: the current script contains a user-specific local directory. Then run the training script and launch the interface:

```bash
python train_model.py
streamlit run streamlit_app.py
```

### Reproducibility note
The current Streamlit code passes the planned delay as a placeholder for the actual delay field. Review the training/prediction feature contract and remove target leakage before using this model for a real evaluation. No performance metric is claimed in this README.

## 🇫🇷 Français

### Présentation
Le dépôt regroupe un module de prétraitement, un script d'entraînement, une fonction de prédiction et une interface Streamlit. Le script d'entraînement construit un pipeline scikit-learn combinant prétraitement et **Random Forest**, avec recherche d'hyperparamètres par **GridSearchCV**.

### Variables saisies dans l'application
- Distance en kilomètres
- Zone de livraison
- Jour de la semaine
- Heure de départ
- Nombre de colis
- Météo ou état du trafic
- Délai prévu

L'interface affiche une classe (« dans les temps » ou « retard ») ainsi qu'une probabilité fournie par le module de prédiction.

### Fichiers
| Fichier | Rôle |
|---|---|
| `streamlit_app.py` | Interface utilisateur |
| `train_model.py` | Entraînement du pipeline et recherche d'hyperparamètres |
| `preprocessing.py` | Prétraitement des variables |
| `predict.py` | Prédiction sur un enregistrement |
| `mini_projet_logistique.csv` | Jeu de données |
| `requirements.txt` | Dépendances Python |

### Installation
```bash
git clone https://github.com/Slymanidiallo/Pr-diction-des-d-lais-de-livraisons.git
cd Pr-diction-des-d-lais-de-livraisons
python -m venv .venv
# Windows : .venv\Scripts\activate
# macOS/Linux : source .venv/bin/activate
pip install -r requirements.txt
```

Avant l'entraînement, corrigez les chemins de données et de sortie dans `train_model.py`, car le script contient actuellement un chemin local propre à un poste. Lancez ensuite :

```bash
python train_model.py
streamlit run streamlit_app.py
```

### Point de vigilance
L'interface transmet actuellement le délai prévu comme valeur provisoire du délai réel. Il faut vérifier le contrat des variables entre entraînement et prédiction et éliminer toute fuite de la variable cible avant une évaluation fiable. Aucune métrique de performance n'est annoncée ici.
