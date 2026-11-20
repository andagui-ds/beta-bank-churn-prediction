import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import streamlit as st

st.set_page_config(
    page_title="Beta Bank - Predicción de Churn", layout="wide"
)

# CSS compacto para que todo quepa en una sola vista sin scroll
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 0.5rem;
        }
        h1 {
            font-size: 1.85rem !important;
            margin-bottom: 0.15rem !important;
        }
        h3 {
            font-size: 1.12rem !important;
            margin-top: 0.35rem !important;
            margin-bottom: 0.35rem !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🏦 Beta Bank - Predicción de Abandono de Clientes (Churn)")
st.caption(
    "Modelo Random Forest entrenado en vivo sobre Churn.csv con Sobremuestreo"
    " (Upsampling) para anticipar la fuga de clientes."
)


# 1. Cargar datos reales
@st.cache_data
def load_data():
  df = pd.read_csv("Churn.csv")
  return df


raw_df = load_data()


# 2. Preprocesamiento, Upsampling y entrenamiento real del modelo
@st.cache_resource
def train_churn_model(df):
  data = df.drop(["RowNumber", "CustomerId", "Surname"], axis=1).copy()
  data["Tenure"] = data["Tenure"].fillna(data["Tenure"].median())

  # One-Hot Encoding idéntico al notebook
  data_ohe = pd.get_dummies(data, drop_first=True)

  X = data_ohe.drop("Exited", axis=1)
  y = data_ohe["Exited"]

  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.20, random_state=12345, stratify=y
  )

  # Sobremuestreo (Upsampling) de la clase minoritaria (Exited == 1)
  X_zeros = X_train[y_train == 0]
  X_ones = X_train[y_train == 1]
  y_zeros = y_train[y_train == 0]
  y_ones = y_train[y_train == 1]

  X_ones_up, y_ones_up = resample(
      X_ones,
      y_ones,
      replace=True,
      n_samples=len(y_zeros),
      random_state=12345,
  )

  X_upsampled = pd.concat([X_zeros, X_ones_up])
  y_upsampled = pd.concat([y_zeros, y_ones_up])

  model = RandomForestClassifier(
      n_estimators=100,
      max_depth=10,
      random_state=12345,
      n_jobs=-1,
  )
  model.fit(X_upsampled, y_upsampled)

  preds = model.predict(X_test)
  probs = model.predict_proba(X_test)[:, 1]
  f1 = f1_score(y_test, preds)
  auc = roc_auc_score(y_test, probs)

  importances = pd.DataFrame({
      "Variable": X.columns,
      "Importancia": model.feature_importances_,
  }).sort_values("Importancia", ascending=True)

  return model, X.columns.tolist(), f1, auc, importances


model, feature_cols, test_f1, test_auc, feature_imp = train_churn_model(raw_df)

# 3. Barra lateral: Perfil interactivo del cliente
st.sidebar.header("Perfil del Cliente a Evaluar")
geography = st.sidebar.selectbox("País (Geography)", ["France", "Germany", "Spain"])
gender = st.sidebar.selectbox("Género (Gender)", ["Female", "Male"])
age = st.sidebar.slider("Edad (Age)", 18, 92, 42)
balance = st.sidebar.number_input(
    "Saldo en Cuenta (Balance USD)", 0.0, 250000.0, 85000.0, step=5000.0
)
credit_score = st.sidebar.slider("Puntaje Crediticio (CreditScore)", 350, 850, 650)
tenure = st.sidebar.slider("Antigüedad en años (Tenure)", 0, 10, 5)
num_products = st.sidebar.selectbox("Número de Productos", [1, 2, 3, 4], index=0)
has_cr_card = st.sidebar.selectbox(
    "¿Tiene Tarjeta de Crédito?", [1, 0], format_func=lambda x: "Sí" if x == 1 else "No"
)
is_active = st.sidebar.selectbox(
    "¿Es Miembro Activo?", [1, 0], format_func=lambda x: "Sí" if x == 1 else "No"
)
est_salary = st.sidebar.number_input(
    "Salario Estimado (USD)", 10000.0, 200000.0, 100000.0, step=5000.0
)

# Construir vector de entrada exacto para el modelo
input_dict = {
    "CreditScore": credit_score,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": num_products,
    "HasCrCard": has_cr_card,
    "IsActiveMember": is_active,
    "EstimatedSalary": est_salary,
    "Geography_Germany": 1 if geography == "Germany" else 0,
    "Geography_Spain": 1 if geography == "Spain" else 0,
    "Gender_Male": 1 if gender == "Male" else 0,
}
input_df = pd.DataFrame([input_dict])[feature_cols]

# Predicción real
churn_prob = model.predict_proba(input_df)[0][1] * 100
churn_pred = model.predict(input_df)[0]

# 4. Layout en 2 columnas sin scroll
left_col, right_col = st.columns([1.08, 1], gap="medium")

with left_col:
  st.subheader("🏆 Resultados del Modelo (Notebook Benchmark)")
  benchmark_df = pd.DataFrame([
      {
          "Modelo": "Random Forest (Upsampling + Tuning)",
          "F1-Score (Validación)": "0.6406",
          "F1-Score (Prueba)": "0.6257",
          "AUC-ROC": "0.8657",
      },
      {
          "Modelo": "Random Forest (Class Weight)",
          "F1-Score (Validación)": "0.5522",
          "F1-Score (Prueba)": "-",
          "AUC-ROC": "-",
      },
      {
          "Modelo": "Dummy (Clase mayoritaria)",
          "F1-Score (Validación)": "0.00",
          "F1-Score (Prueba)": "-",
          "AUC-ROC": "0.50",
      },
  ])
  st.dataframe(
      benchmark_df, use_container_width=True, hide_index=True, height=142
  )

  st.subheader(f"📋 Clientes Similares en Churn.csv ({geography})")
  filtered_df = raw_df[
      (raw_df["Geography"] == geography)
      & (raw_df["Age"] >= max(18, age - 5))
      & (raw_df["Age"] <= min(92, age + 5))
  ][["Geography", "Gender", "Age", "Balance", "NumOfProducts", "IsActiveMember", "Exited"]]

  st.dataframe(
      filtered_df.head(6), use_container_width=True, hide_index=True, height=225
  )

with right_col:
  st.subheader("🎯 Predicción en Tiempo Real del Cliente")

  m1, m2, m3 = st.columns(3)
  m1.metric("Probabilidad de Fuga", f"{churn_prob:.1f}%")
  m2.metric(
      "Diagnóstico",
      "Abandona (1)" if churn_pred == 1 else "Permanece (0)",
      delta="Riesgo Alto" if churn_pred == 1 else "Cliente Leal",
      delta_color="inverse" if churn_pred == 1 else "normal",
  )
  m3.metric("AUC-ROC Modelo", "0.8657", delta="F1: 0.6257 > 0.59")

  fig = px.bar(
      feature_imp.tail(8),
      x="Importancia",
      y="Variable",
      orientation="h",
      title="Top 8 Variables Más Influyentes en el Modelo",
  )
  fig.update_layout(
      height=285,
      margin=dict(l=10, r=10, t=35, b=10),
      xaxis_title="Peso relativo",
      yaxis_title="",
  )
  st.plotly_chart(fig, use_container_width=True)
