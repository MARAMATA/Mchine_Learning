import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Fonction pour traiter les valeurs aberrantes (outliers)
def treat_outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = df[col].apply(lambda x: lower_bound if x < lower_bound else upper_bound if x > upper_bound else x)
    return df

# Fonction de prétraitement complet avec pipeline
def preprocess_data(data):
    # Séparation des variables numériques et catégorielles
    numerical_features = data.select_dtypes(include=['float64', 'int64']).columns
    categorical_features = data.select_dtypes(include=['object']).columns

    # Pipeline pour les variables numériques
    numerical_pipeline = Pipeline(steps=[ 
        ('imputer', SimpleImputer(strategy='median')),  # Imputation des valeurs manquantes
        ('scaler', StandardScaler())  # Normalisation des données
    ])

    # Pipeline pour les variables catégorielles
    categorical_pipeline = Pipeline(steps=[ 
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Imputation avec la valeur la plus fréquente
        ('onehot', OneHotEncoder(handle_unknown='ignore'))  # Encodage OneHot
    ])

    # Appliquer le pipeline aux données
    preprocessor = ColumnTransformer(transformers=[ 
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ])

    # Appliquer la transformation
    data_preprocessed = preprocessor.fit_transform(data)
    
    # Retourner les données prétraitées sous forme de DataFrame
    columns = list(numerical_features) + list(preprocessor.transformers_[1][1].named_steps['onehot'].get_feature_names_out(categorical_features))
    data_preprocessed_df = pd.DataFrame(data_preprocessed, columns=columns)

    # Traitement des outliers
    data_preprocessed_df = treat_outliers(data_preprocessed_df, numerical_features)
    
    return data_preprocessed_df

# Fonction pour charger les données
def load_data(uploaded_file=None):
    # Si aucun fichier n'est téléchargé, charger le fichier par défaut
    if uploaded_file is None:
        default_file = 'Invistico_Airline.csv'
        if os.path.exists(default_file):
            return pd.read_csv(default_file)
        else:
            st.error(f"Le fichier {default_file} n'a pas été trouvé.")
            return None
    else:
        try:
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                data = pd.read_excel(uploaded_file)
            else:
                st.error("Format de fichier non pris en charge.")
                return None
            return data
        except Exception as e:
            st.error(f"Erreur lors du chargement du fichier : {e}")
            return None

# Fonction pour effectuer des prédictions
def predict(data, model_choice):
    X = data.drop(columns=['satisfaction_satisfied'])
    y = data['satisfaction_satisfied']
    
    # Séparer les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Choisir le modèle
    if model_choice == 'Régression Linéaire':
        model = LinearRegression()
    elif model_choice == 'K-Nearest Neighbors (KNN)':
        model = KNeighborsClassifier(n_neighbors=5)
    else:
        model = GaussianNB()

    # Entraîner le modèle
    model.fit(X_train, y_train)

    # Initialiser la variable y_pred_class
    y_pred_class = None
    proba = None

    # Prédictions
    if hasattr(model, 'predict_proba'):  # Si le modèle prédit des probabilités
        proba = model.predict_proba(X_test)[:, 1]  # Probabilités pour la classe positive (classe 1)
        y_pred_class = (proba >= 0.5).astype(int)  # Convertir les probabilités en classes binaires
    else:  # Si le modèle ne prédit pas des probabilités (comme la régression linéaire)
        y_pred = model.predict(X_test)
        y_pred_class = (y_pred >= 0.5).astype(int)  # Convertir les scores de régression en classes binaires
        proba = y_pred_class  # Utiliser directement les prédictions binaires

    # Calcul des métriques de classification
    precision = precision_score(y_test, y_pred_class)
    recall = recall_score(y_test, y_pred_class)
    f1 = f1_score(y_test, y_pred_class)

    return proba, precision, recall, f1

# Interface utilisateur
st.title("Prédiction de la Satisfaction Client")

# Choix entre téléchargement de fichier ou saisie manuelle
input_option = st.selectbox("Choisissez une méthode d'entrée", ["Télécharger un fichier", "Saisir manuellement"])

# Charger les données
data = None
if input_option == "Télécharger un fichier":
    uploaded_file = st.file_uploader("Téléchargez un fichier CSV ou Excel", type=["csv", "xlsx", "xls"])
    data = load_data(uploaded_file)  # Charge soit le fichier par défaut, soit un fichier téléchargé

    if data is not None:
        st.write("Données chargées avec succès.")
        st.write(data.head())
        # Prétraiter les données
        data = preprocess_data(data)
        st.write("Données après prétraitement")
        st.write(data.head())

elif input_option == "Saisir manuellement":
    st.write("Entrez les données manuellement :")
    feature_1 = st.number_input("Feature 1 (ex: Age)")
    feature_2 = st.number_input("Feature 2 (ex: Type de voyage)")
    feature_3 = st.number_input("Feature 3 (ex: Distance)")
    feature_4 = st.number_input("Feature 4 (ex: Temps de retard)")
    
    new_data = pd.DataFrame({
        'Feature 1': [feature_1],
        'Feature 2': [feature_2],
        'Feature 3': [feature_3],
        'Feature 4': [feature_4]
    })

    st.write("Données saisies :")
    st.write(new_data)

    data = new_data

# Choisir le modèle
model_choice = st.selectbox("Sélectionnez le modèle pour prédiction", ["Régression Linéaire", "K-Nearest Neighbors (KNN)", "Naïve Bayes"])

# Prédictions
if st.button("Faire la Prédiction"):
    if data is not None:
        # Effectuer la prédiction
        proba, precision, recall, f1 = predict(data, model_choice)

        # Afficher les résultats
        st.write(f"**Probabilité de Satisfaction Prédite avec {model_choice}:**")
        st.write(f"Probabilité : {proba.mean():.2f}")
        st.write(f"Précision : {precision:.2f}, Rappel : {recall:.2f}, F1-score : {f1:.2f}")

        # Visualiser les résultats
        st.write("Graphique des probabilités de satisfaction :")
        fig, ax = plt.subplots()
        ax.hist(proba, bins=20, color='blue', alpha=0.7)
        ax.set_title(f"Distribution des Probabilités de Satisfaction - {model_choice}")
        ax.set_xlabel('Probabilité')
        ax.set_ylabel('Fréquence')
        st.pyplot(fig)
