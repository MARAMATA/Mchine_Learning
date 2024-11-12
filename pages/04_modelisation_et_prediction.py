import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import pandas as pd

st.title("Modélisation et Prédiction de la Satisfaction Client")

# Vérifier si les données prétraitées sont dans `st.session_state`
if 'data' in st.session_state:
    data = st.session_state['data']
else:
    st.error("Erreur : Les données prétraitées ne sont pas chargées.")
    st.stop()

# Sélection des caractéristiques et de la variable cible
X = data.drop(columns=['satisfaction_satisfied'])  # Variables explicatives
y = data['satisfaction_satisfied']  # Variable cible (satisfaction)

# Diviser les données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- Régression Linéaire ---
st.header("Régression Linéaire")

# Initialisation du modèle de régression linéaire
linreg = LinearRegression()

# Entraînement du modèle
linreg.fit(X_train, y_train)

# Prédictions
y_pred_linreg = linreg.predict(X_test)

# Calcul de la précision, rappel et F1-score pour la régression (en utilisant une classification binaire)
y_pred_class_linreg = [1 if i >= 0.5 else 0 for i in y_pred_linreg]
precision_linreg = precision_score(y_test, y_pred_class_linreg)
recall_linreg = recall_score(y_test, y_pred_class_linreg)
f1_linreg = f1_score(y_test, y_pred_class_linreg)

# Sauvegarde des résultats dans `st.session_state`
st.session_state['linreg_results'] = {'precision': precision_linreg, 'recall': recall_linreg, 'f1': f1_linreg}

# Affichage des performances de la régression linéaire
with st.expander("Voir les performances de la Régression Linéaire"):
    st.write(f"Précision : {precision_linreg:.2f}")
    st.write(f"Rappel : {recall_linreg:.2f}")
    st.write(f"F1-score : {f1_linreg:.2f}")

# --- K-Nearest Neighbors (KNN) ---
st.header("K-Nearest Neighbors (KNN)")

# Initialisation du modèle KNN
knn = KNeighborsClassifier(n_neighbors=5)

# Entraînement du modèle
knn.fit(X_train, y_train)

# Prédictions
y_pred_knn = knn.predict(X_test)

# Calcul de la précision, rappel et F1-score pour KNN
precision_knn = precision_score(y_test, y_pred_knn)
recall_knn = recall_score(y_test, y_pred_knn)
f1_knn = f1_score(y_test, y_pred_knn)

# Sauvegarde des résultats KNN avant optimisation
st.session_state['knn_results'] = {'precision': precision_knn, 'recall': recall_knn, 'f1': f1_knn}

# Affichage des performances de KNN
with st.expander("Voir les performances de KNN"):
    st.write(f"Précision : {precision_knn:.2f}")
    st.write(f"Rappel : {recall_knn:.2f}")
    st.write(f"F1-score : {f1_knn:.2f}")

# --- Naïve Bayes ---
st.header("Naïve Bayes")

# Initialisation du modèle Naïve Bayes
nb = GaussianNB()

# Entraînement du modèle
nb.fit(X_train, y_train)

# Prédictions
y_pred_nb = nb.predict(X_test)

# Calcul des scores pour Naïve Bayes
precision_nb = precision_score(y_test, y_pred_nb)
recall_nb = recall_score(y_test, y_pred_nb)
f1_nb = f1_score(y_test, y_pred_nb)

# Sauvegarde des résultats Naïve Bayes
st.session_state['nb_results'] = {'precision': precision_nb, 'recall': recall_nb, 'f1': f1_nb}

# Affichage des performances de Naïve Bayes
with st.expander("Voir les performances de Naïve Bayes"):
    st.write(f"Précision : {precision_nb:.2f}")
    st.write(f"Rappel : {recall_nb:.2f}")
    st.write(f"F1-score : {f1_nb:.2f}")

# --- Comparaison des Modèles ---
st.header("Comparaison des Modèles")

# Créer un DataFrame pour les résultats de comparaison
comparison_df = pd.DataFrame({
    "Modèle": ["Régression Linéaire", "KNN", "Naïve Bayes"],
    "Précision": [precision_linreg, precision_knn, precision_nb],
    "Rappel": [recall_linreg, recall_knn, recall_nb],
    "F1-score": [f1_linreg, f1_knn, f1_nb]
})

# Affichage de la comparaison sous forme de tableau
st.dataframe(comparison_df)

# Affichage du rapport comparatif
st.write("""
### Rapport de Comparaison des Modèles


KNN est le modèle le plus performant avec une précision de 0.94, un rappel de 0.91, et un F1-score de 0.93. Il est particulièrement adapté pour ce type de problème de classification binaire.

Régression Linéaire, bien que fonctionnelle, n'est pas aussi efficace dans cette tâche de classification. Cependant, elle reste un bon modèle de base.

Naïve Bayes a des performances plus faibles, avec un rappel et une précision de 0.83, ce qui le rend moins performant que KNN pour cette tâche.

En conclusion, KNN est recommandé pour ce cas d'utilisation, mais la régression linéaire peut être utilisée pour une analyse exploratoire initiale, tandis que Naïve Bayes peut être un modèle rapide mais moins précis.
""")
