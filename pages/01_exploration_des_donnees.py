import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Exploration des Données - Satisfaction Client")

# Accéder aux données stockées dans `st.session_state`
if 'data' in st.session_state:
    data = st.session_state['data']
else:
    st.error("Erreur : Les données ne sont pas chargées.")
    st.stop()

# Affichage des premières lignes du dataset
st.header("Aperçu des Données")
st.write("Visualisez les premières lignes de notre dataset pour un aperçu initial.")
st.write(data.head())

# Création des colonnes pour l'affichage côte à côte
st.header("Types de Données et Valeurs Manquantes")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Types de Données")
    st.write(data.dtypes)

with col2:
    st.subheader("Valeurs Manquantes")
    st.write(data.isnull().sum())

# Statistiques descriptives
st.header("Statistiques Descriptives")
st.write(data.describe())

# Distribution des classes de satisfaction
st.header("Distribution des Classes")
fig, ax = plt.subplots()
sns.countplot(x="satisfaction", data=data, ax=ax)
ax.set_title("Distribution des Clients Satisfaits et Non-Satisfaits")
st.pyplot(fig)

# Matrice de corrélation
st.header("Matrice de Corrélation")
numeric_data = data.select_dtypes(include=['float64', 'int64'])  # Sélectionner seulement les colonnes numériques
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", ax=ax)
ax.set_title("Matrice de Corrélation des Variables Numériques")
st.pyplot(fig)

# Boxplot pour chaque variable numérique
st.header("Boxplots pour Détecter les Outliers")

# Organisation des boxplots en deux colonnes
num_cols = numeric_data.columns
col1, col2 = st.columns(2)

for i, col in enumerate(num_cols):
    with col1 if i % 2 == 0 else col2:
        fig, ax = plt.subplots()
        sns.boxplot(y=numeric_data[col], ax=ax)
        ax.set_title(f"Boxplot de {col}")
        st.pyplot(fig)



