import streamlit as st

from predict import predict_single


st.set_page_config(page_title="Delivery Delay Prediction", layout="centered")
st.title("📦 Prédiction des délais de livraison")
st.write("Estimez si une livraison sera effectuée dans les temps ou avec retard.")

with st.form("prediction_form"):
    distance_km = st.number_input("Distance (km)", min_value=0.1, value=10.0)
    zone = st.selectbox("Zone", ["urbaine", "périurbaine", "rurale"])
    jour_semaine = st.selectbox(
        "Jour de la semaine",
        ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"],
    )
    heure_depart = st.selectbox("Heure de départ", ["matin", "après-midi", "soir"])
    nb_colis = st.number_input("Nombre de colis", min_value=1, value=5)
    meteo = st.selectbox("Météo / trafic", ["ensoleillé", "pluie", "embouteillage"])
    delai_prevu = st.number_input("Délai prévu (min)", min_value=1.0, value=30.0)
    submitted = st.form_submit_button("Prédire")

if submitted:
    features = {
        "distance_km": distance_km,
        "zone": zone,
        "jour_semaine": jour_semaine,
        "heure_depart": heure_depart,
        "nb_colis": nb_colis,
        "meteo": meteo,
        "delai_prevu": delai_prevu,
    }
    try:
        prediction, delay_probability = predict_single(features)
        status = "🚀 Dans les temps" if prediction == 0 else "⚠️ Retard"
        st.subheader(f"Résultat : {status}")
        st.metric("Probabilité estimée de retard", f"{delay_probability:.1%}")
    except FileNotFoundError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
