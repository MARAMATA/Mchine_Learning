import streamlit as st
from sklearn.preprocessing import StandardScaler
import pandas as pd

st.title("Prétraitement des Données")

# Affichage du code de prétraitement pour documentation
st.header("Code de Prétraitement Utilisé")
code = '''
import streamlit as st
from sklearn.preprocessing import StandardScaler
import pandas as pd

st.title("Prétraitement des Données")

# Accéder aux données stockées dans `st.session_state`
if 'data' in st.session_state:
    data = st.session_state['data']
else:
    st.error("Erreur : Les données ne sont pas chargées.")
    st.stop()

# Imputation des valeurs manquantes
st.header("Imputation des Valeurs Manquantes")

# Imputer les colonnes numériques avec la médiane
numerical_columns = data.select_dtypes(include=["float64", "int64"]).columns
data[numerical_columns] = data[numerical_columns].fillna(data[numerical_columns].median())

# Imputer les colonnes catégorielles avec la valeur la plus fréquente
categorical_columns = data.select_dtypes(include=["object"]).columns
data[categorical_columns] = data[categorical_columns].apply(lambda x: x.fillna(x.mode()[0]))

# Afficher les valeurs manquantes restantes pour vérifier que l'imputation a été faite
st.write("Colonnes avec des valeurs manquantes après imputation :")
st.write(data.isnull().sum()[data.isnull().sum() > 0])

# Conversion de la colonne 'Arrival Delay in Minutes' en int64 après imputation
if 'Arrival Delay in Minutes' in data.columns:
    data['Arrival Delay in Minutes'] = data['Arrival Delay in Minutes'].astype('int64')
    st.write("Valeurs manquantes après conversion de 'Arrival Delay in Minutes' :", data['Arrival Delay in Minutes'].isnull().sum())

# Afficher la colonne 'Arrival Delay in Minutes' après prétraitement
st.write("### Colonne 'Arrival Delay in Minutes' après Prétraitement")
st.write(data['Arrival Delay in Minutes'].head(10))  # Afficher les 10 premières valeurs

# Traitement des Outliers
outlier_columns = ['On-board service', 'Checkin service']

def treat_outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Remplacer les valeurs en dehors des bornes par les limites
        df[col] = df[col].apply(lambda x: lower_bound if x < lower_bound else upper_bound if x > upper_bound else x)
        
    return df

# Appliquer le traitement des outliers aux colonnes spécifiées
data = treat_outliers(data, outlier_columns)

# Afficher les données après traitement des outliers
st.write("### Données après traitement des Outliers pour les colonnes 'On-board service' et 'Checkin service'")
st.write(data[outlier_columns].describe())

# Encodage des variables catégorielles
st.header("Encodage des Variables Catégorielles")
data = pd.get_dummies(data, drop_first=True)

# Standardisation des données numériques
st.header("Standardisation des Données")
scaler = StandardScaler()
numerical_features = data.select_dtypes(include=["float64", "int64"]).columns
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Mettre à jour `st.session_state['data']` avec les données prétraitées
st.session_state['data'] = data

st.write("### Données après Prétraitement")
st.write(data.head())
'''
st.code(code, language='python')

# Accéder aux données stockées dans `st.session_state`
if 'data' in st.session_state:
    data = st.session_state['data']
else:
    st.error("Erreur : Les données ne sont pas chargées.")
    st.stop()

# Imputation des valeurs manquantes
st.header("Imputation des Valeurs Manquantes")

# Imputer les colonnes numériques avec la médiane
numerical_columns = data.select_dtypes(include=["float64", "int64"]).columns
data[numerical_columns] = data[numerical_columns].fillna(data[numerical_columns].median())

# Imputer les colonnes catégorielles avec la valeur la plus fréquente
categorical_columns = data.select_dtypes(include=["object"]).columns
data[categorical_columns] = data[categorical_columns].apply(lambda x: x.fillna(x.mode()[0]))

# Afficher les valeurs manquantes restantes pour vérifier que l'imputation a été faite
st.write("Colonnes avec des valeurs manquantes après imputation :")
st.write(data.isnull().sum()[data.isnull().sum() > 0])

# Conversion de la colonne 'Arrival Delay in Minutes' en int64 après imputation
if 'Arrival Delay in Minutes' in data.columns:
    data['Arrival Delay in Minutes'] = data['Arrival Delay in Minutes'].astype('int64')
    st.write("Valeurs manquantes après conversion de 'Arrival Delay in Minutes' :", data['Arrival Delay in Minutes'].isnull().sum())

# Afficher la colonne 'Arrival Delay in Minutes' après prétraitement
st.write("### Colonne 'Arrival Delay in Minutes' après Prétraitement")
st.write(data['Arrival Delay in Minutes'].head(10))  # Afficher les 10 premières valeurs

# Traitement des Outliers
outlier_columns = ['On-board service', 'Checkin service']

def treat_outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Remplacer les valeurs en dehors des bornes par les limites
        df[col] = df[col].apply(lambda x: lower_bound if x < lower_bound else upper_bound if x > upper_bound else x)
        
    return df

# Appliquer le traitement des outliers aux colonnes spécifiées
data = treat_outliers(data, outlier_columns)

# Afficher les données après traitement des outliers
st.write("### Données après traitement des Outliers pour les colonnes 'On-board service' et 'Checkin service'")
st.write(data[outlier_columns].describe())

# Encodage des variables catégorielles
st.header("Encodage des Variables Catégorielles")
data = pd.get_dummies(data, drop_first=True)

# Standardisation des données numériques
st.header("Standardisation des Données")
scaler = StandardScaler()
numerical_features = data.select_dtypes(include=["float64", "int64"]).columns
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Mettre à jour `st.session_state['data']` avec les données prétraitées
st.session_state['data'] = data

st.write("### Données après Prétraitement")
st.write(data.head())




