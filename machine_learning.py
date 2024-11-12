import streamlit as st
import pandas as pd

# Chargement des données avec mise en cache pour optimiser les performances
@st.cache_data
def load_data():
    data = pd.read_csv('Invistico_Airline.csv')
    return data

# Charger les données une fois
data = load_data()

# Rendre les données disponibles dans l'ensemble de l'application
st.session_state['data'] = data


import streamlit as st

# Titre de l'application
st.title("Prédiction de la Satisfaction Client")

# Explication pour naviguer dans les différentes pages
st.write("""
Bienvenue sur l'application de prédiction de la satisfaction client. 
Naviguez dans les différentes pages pour explorer les étapes du projet, de l'exploration des données à la modélisation, l'évaluation et la visualisation interactive.
""")
