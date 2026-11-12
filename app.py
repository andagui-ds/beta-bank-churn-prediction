import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# 1. Título de la app
st.title("🏦 Beta Bank - Predicción de Churn")

# 2. Carga de datos (puedes usar @st.cache_data para optimizar)


@st.cache_data
def load_data():
    # Asegúrate de que el archivo esté en la carpeta
    df = pd.read_csv('Churn.csv')
    return df


df = load_data()

# 3. Widgets interactivos (ej. sliders para que el usuario explore)
st.sidebar.header("Filtros de Cliente")
age = st.sidebar.slider("Edad", 18, 100, 40)
balance = st.sidebar.number_input("Balance", 0.0, 250000.0, 50000.0)

# 4. Visualización de datos
st.subheader("Datos Filtrados")
filtered_df = df[(df['Age'] >= age) & (df['Balance'] >= balance)]
st.dataframe(filtered_df.head())

# 5. Botón para predecir (simulado)
if st.button("Predecir Churn"):
    st.success("El cliente tiene una alta probabilidad de irse.")
