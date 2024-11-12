import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd

# Récupérer les données prétraitées
if 'data' in st.session_state:
    data = st.session_state['data']
else:
    st.error("Erreur : Les données prétraitées ne sont pas chargées.")
    st.stop()

# Sélection des caractéristiques et de la variable cible
X = data.drop(columns=['satisfaction_satisfied'])
y = data['satisfaction_satisfied']

# Diviser les données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- Évaluation avant optimisation ---
st.header("Évaluation Avant Optimisation")

# Récupérer les résultats sauvegardés des performances avant optimisation
linreg_results = st.session_state.get('linreg_results', None)
knn_results = st.session_state.get('knn_results', None)
nb_results = st.session_state.get('nb_results', None)

# Affichage des résultats avant optimisation
if linreg_results:
    st.write("**Régression Linéaire** :")
    st.write(f"Précision : {linreg_results['precision']:.2f}, Rappel : {linreg_results['recall']:.2f}, F1-score : {linreg_results['f1']:.2f}")

if knn_results:
    st.write("**K-Nearest Neighbors (KNN)** :")
    st.write(f"Précision : {knn_results['precision']:.2f}, Rappel : {knn_results['recall']:.2f}, F1-score : {knn_results['f1']:.2f}")

if nb_results:
    st.write("**Naïve Bayes** :")
    st.write(f"Précision : {nb_results['precision']:.2f}, Rappel : {nb_results['recall']:.2f}, F1-score : {nb_results['f1']:.2f}")

# --- Optimisation de tous les modèles ---
st.header("Optimisation des Modèles avec GridSearchCV")

# GridSearch pour KNN
knn_param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}
knn_grid = GridSearchCV(KNeighborsClassifier(), knn_param_grid, cv=5, scoring='accuracy')
knn_grid.fit(X_train, y_train)

# GridSearch pour Naïve Bayes (aucun hyperparamètre à optimiser, donc simple)
nb_grid = GridSearchCV(GaussianNB(), param_grid={}, cv=5, scoring='accuracy')
nb_grid.fit(X_train, y_train)

# GridSearch pour la Régression Linéaire (aucun hyperparamètre à optimiser, donc simple)
linreg_grid = GridSearchCV(LinearRegression(), param_grid={}, cv=5, scoring='accuracy')
linreg_grid.fit(X_train, y_train)

# Résultats de l'optimisation
st.write(f"Meilleur nombre de voisins pour KNN : {knn_grid.best_params_['n_neighbors']}")
st.write(f"Meilleure précision de KNN après optimisation : {knn_grid.best_score_:.2f}")

st.write(f"Meilleure précision de Naïve Bayes après optimisation : {nb_grid.best_score_:.2f}")
st.write(f"Meilleure précision de Régression Linéaire après optimisation : {linreg_grid.best_score_:.2f}")

# --- Comparaison des Performances Avant et Après Optimisation ---
st.header("Comparaison des Performances Avant et Après Optimisation")

# Calcul des scores optimisés pour KNN
y_pred_knn_optimized = knn_grid.predict(X_test)
precision_knn_optimized = precision_score(y_test, y_pred_knn_optimized)
recall_knn_optimized = recall_score(y_test, y_pred_knn_optimized)
f1_knn_optimized = f1_score(y_test, y_pred_knn_optimized)

# Calcul des scores optimisés pour Naïve Bayes
y_pred_nb_optimized = nb_grid.predict(X_test)
precision_nb_optimized = precision_score(y_test, y_pred_nb_optimized)
recall_nb_optimized = recall_score(y_test, y_pred_nb_optimized)
f1_nb_optimized = f1_score(y_test, y_pred_nb_optimized)

# Calcul des scores optimisés pour Régression Linéaire
y_pred_linreg_optimized = linreg_grid.predict(X_test)
y_pred_class_linreg_optimized = [1 if i >= 0.5 else 0 for i in y_pred_linreg_optimized]
precision_linreg_optimized = precision_score(y_test, y_pred_class_linreg_optimized)
recall_linreg_optimized = recall_score(y_test, y_pred_class_linreg_optimized)
f1_linreg_optimized = f1_score(y_test, y_pred_class_linreg_optimized)

# Créer un DataFrame pour comparer les résultats
comparison_df_optimized = pd.DataFrame({
    "Modèle": ["KNN avant optimisation", "KNN après optimisation", "Naïve Bayes avant optimisation", "Naïve Bayes après optimisation", "Régression Linéaire avant optimisation", "Régression Linéaire après optimisation"],
    "Précision": [knn_results['precision'], precision_knn_optimized, nb_results['precision'], precision_nb_optimized, linreg_results['precision'], precision_linreg_optimized],
    "Rappel": [knn_results['recall'], recall_knn_optimized, nb_results['recall'], recall_nb_optimized, linreg_results['recall'], recall_linreg_optimized],
    "F1-score": [knn_results['f1'], f1_knn_optimized, nb_results['f1'], f1_nb_optimized, linreg_results['f1'], f1_linreg_optimized]
})

# Affichage de la comparaison
st.dataframe(comparison_df_optimized)




