import streamlit as st
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Réduction de Dimensionnalité avec l'ACP")
st.write("Page de réduction de dimensionnalité avec l’ACP. Appliquer l’ACP pour réduire les dimensions et visualiser les données projetées pour explorer les groupes et les clusters éventuels.")

# Vérifier si les données prétraitées existent dans `st.session_state`
if 'data' in st.session_state:
    data = st.session_state['data']
else:
    st.error("Erreur : Les données prétraitées ne sont pas chargées.")
    st.stop()
    
# Vérifier si la colonne 'satisfaction_satisfied' existe
if 'satisfaction_satisfied' not in data.columns:
    st.error("Erreur : La colonne 'satisfaction_satisfied' est manquante dans les données.")
    st.stop()

# Sélection des caractéristiques et de la variable cible (satisfaction_satisfied)
X = data.drop(columns=['satisfaction_satisfied'])  # Caractéristiques
y = data['satisfaction_satisfied']  # Variable cible

# Application de l'ACP pour réduire les dimensions à 2
st.header("Application de l'ACP")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Création d'un DataFrame pour les résultats de l'ACP
pca_df = pd.DataFrame(X_pca, columns=['Composante 1', 'Composante 2'])
pca_df['Satisfaction'] = y.values

# Visualisation des données projetées en 2D
st.header("Visualisation en 2D des Données après ACP")
fig, ax = plt.subplots(figsize=(10, 8))
sns.scatterplot(x='Composante 1', y='Composante 2', hue='Satisfaction', data=pca_df, palette="viridis", ax=ax)
ax.set_xlabel('Composante 1')
ax.set_ylabel('Composante 2')
ax.set_title("Projection des Données en 2D avec l'ACP", fontsize=15)
st.pyplot(fig)

# Interprétation des composantes principales
st.header("Interprétation des Composantes Principales")
explained_variance = pca.explained_variance_ratio_
st.write("Variance expliquée par la première composante :", f"{explained_variance[0]:.2%}")
st.write("Variance expliquée par la deuxième composante :", f"{explained_variance[1]:.2%}")

# Affichage des variables influençant le plus les composantes principales
components = pd.DataFrame(pca.components_, columns=X.columns, index=['Composante 1', 'Composante 2']).T
st.write("Contribution des Variables aux Composantes Principales")
st.write(components)

# Interprétation finale
st.markdown("""
### Interprétation
Interprétation des Composantes Principales :

Composante 1 :

Elle est dominée par des facteurs physiques de confort et de service en vol :

Baggage handling (0.2715)

Cleanliness (0.2744)

Leg room service (0.2449)

Seat comfort (0.2249)

Ces variables indiquent que la qualité du service (gestion des bagages, confort des sièges, espace pour les jambes, propreté) a une influence majeure sur la satisfaction globale des passagers. Composante 1 explique 20.86% de la variance, soulignant l'importance de ces éléments tangibles dans l'expérience client.

Composante 2 :

Elle reflète des aspects numériques et de commodité, tels que :

Gate location (0.4568)

Ease of Online booking (0.4189)

Inflight wifi service (0.2815)

Online support (0.3370)

Ces variables mettent en lumière l'impact de l’expérience numérique et de la facilité d'embarquement sur la satisfaction des passagers. Composante 2 explique 13.43% de la variance, capturant les éléments intangibles de l’expérience.

Conclusion :
            
Composante 1 : Confort physique et services en vol, essentiels pour la satisfaction.
Composante 2 : Qualité de l'expérience numérique, influençant la perception de la facilité et de la connectivité.
            
Les deux composantes montrent que les services physiques et les services numériques jouent chacun un rôle clé dans la satisfaction des passagers, mais dans des dimensions différentes de l’expérience de vol.""")




