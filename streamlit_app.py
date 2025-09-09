import streamlit as st
import pandas as pd
from predict import predict_single  # <- import corrigé si exécuté depuis app/

st.set_page_config(page_title="📦 Prédiction délais de livraison", layout="centered")

st.title("📦 Prédiction des délais de livraison")
st.write("Cette application prédit si une livraison sera **dans les temps** ou en **retard**.")

# Formulaire de saisie
with st.form("prediction_form"):
    distance_km = st.number_input("Distance (km)", min_value=1, value=10)
    zone = st.selectbox("Zone", ["urbaine", "périurbaine", "rurale"])
    jour_semaine = st.selectbox("Jour de la semaine", ["Lundi","Mardi","Mercredi","Jeudi","Vendredi","Samedi","Dimanche"])
    heure_depart = st.selectbox("Heure de départ", ["matin","après-midi","soir"])
    nb_colis = st.number_input("Nombre de colis", min_value=1, value=5)
    meteo = st.selectbox("Météo", ["ensoleillé", "pluie", "embouteillage"])
    delai_prevu = st.number_input("Délai prévu (min)", min_value=10, value=30)
    
    submitted = st.form_submit_button("Prédire")

# Prédiction
if submitted:
    data = {
        "distance_km": distance_km,
        "zone": zone,
        "jour_semaine": jour_semaine,
        "heure_depart": heure_depart,
        "nb_colis": nb_colis,
        "meteo": meteo,
        "delai_prevu": delai_prevu,
        "delai_reel": delai_prevu  # placeholder
    }
    pred, proba = predict_single(data)
    statut = "🚀 Dans les temps" if pred == 0 else "⚠️ Retard"
    st.subheader(f"Résultat : {statut}")
    st.write(f"Probabilité de retard : {proba:.2f}")
