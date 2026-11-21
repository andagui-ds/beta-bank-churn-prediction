import numpy as np  # Operaciones numéricas y arreglos estadísticos
import pandas as pd  # Manipulación de tablas, codificación One-Hot y filtrado de clientes
# Gráfica interactiva de barras horizontales para importancia de variables
import plotly.express as px
# Clasificador Random Forest para predecir fuga (Churn)
from sklearn.ensemble import RandomForestClassifier
# Métricas de evaluación F1 y AUC-ROC
from sklearn.metrics import f1_score, roc_auc_score
# División estratificada de entrenamiento y prueba
from sklearn.model_selection import train_test_split
# Sobremuestreo (Upsampling) para equilibrar la clase minoritaria
from sklearn.utils import resample
import streamlit as st  # Interfaz web interactiva nativa sin CSS

st.set_page_config(  # Configura el título de la pestaña y el modo ancho (wide)
    page_title="Beta Bank - Predicción de Churn",
    page_icon="🏦",
    layout="wide",
)

# Encabezado compacto nativo
st.header("🏦 Beta Bank - Predicción de Abandono de Clientes (Churn)")
st.caption(  # Subtítulo explicativo del flujo de entrenamiento y balanceo de clases
    "Modelo Random Forest entrenado en vivo sobre Churn.csv con Sobremuestreo"
    " (Upsampling) para anticipar la fuga de clientes."
)


# 1. Cargar datos reales
@st.cache_data  # Guarda el dataset en caché para evitar lecturas repetidas en disco
def load_data():
    # Lee el archivo de clientes desde la raíz del repositorio
    df = pd.read_csv("Churn.csv")
    return df


raw_df = load_data()  # Carga la base histórica de clientes de Beta Bank


# 2. Preprocesamiento, Upsampling y entrenamiento real del modelo
# Mantiene el modelo entrenado en memoria caché para inferencia instantánea
@st.cache_resource
def train_churn_model(df):
    # Elimina identificadores sin valor predictivo
    data = df.drop(["RowNumber", "CustomerId", "Surname"], axis=1).copy()
    # Imputa valores nulos con la mediana
    data["Tenure"] = data["Tenure"].fillna(data["Tenure"].median())

    # Codificación One-Hot evitando multicolinealidad
    data_ohe = pd.get_dummies(data, drop_first=True)

    # Variables explicativas del perfil del cliente
    X = data_ohe.drop("Exited", axis=1)
    # Variable objetivo (1 = abandona el banco, 0 = permanece)
    y = data_ohe["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(  # División estratificada 80% / 20%
        X, y, test_size=0.20, random_state=12345, stratify=y
    )

    # Sobremuestreo (Upsampling) de la clase minoritaria (Exited == 1)
    X_zeros = X_train[y_train == 0]
    X_ones = X_train[y_train == 1]
    y_zeros = y_train[y_train == 0]
    y_ones = y_train[y_train == 1]

    X_ones_up, y_ones_up = resample(  # Duplica registros de fuga hasta igualar la clase mayoritaria
        X_ones,
        y_ones,
        replace=True,
        n_samples=len(y_zeros),
        random_state=12345,
    )

    # Une características balanceadas
    X_upsampled = pd.concat([X_zeros, X_ones_up])
    y_upsampled = pd.concat([y_zeros, y_ones_up])  # Une etiquetas balanceadas

    model = RandomForestClassifier(  # Inicializa Random Forest con los hiperparámetros óptimos del notebook
        n_estimators=100,
        max_depth=10,
        random_state=12345,
        n_jobs=-1,
    )
    # Entrena el modelo sobre el conjunto balanceado
    model.fit(X_upsampled, y_upsampled)

    preds = model.predict(X_test)  # Predicciones binarias en prueba
    # Probabilidades de abandono en prueba
    probs = model.predict_proba(X_test)[:, 1]
    f1 = f1_score(y_test, preds)  # Calcula métrica F1-Score
    auc = roc_auc_score(y_test, probs)  # Calcula métrica AUC-ROC

    importances = pd.DataFrame({  # Extrae la importancia relativa de cada variable
        "Variable": X.columns,
        "Importancia": model.feature_importances_,
    }).sort_values("Importancia", ascending=True)

    return model, X.columns.tolist(), f1, auc, importances


model, feature_cols, test_f1, test_auc, feature_imp = train_churn_model(raw_df)

# 3. Barra lateral: Perfil interactivo del cliente
# Título del panel lateral
st.sidebar.subheader("Perfil del Cliente a Evaluar")

# Agrupa país y género en dos columnas compactas
c_s1, c_s2 = st.sidebar.columns(2)
geography = c_s1.selectbox("País", ["France", "Germany", "Spain"])
gender = c_s2.selectbox("Género", ["Female", "Male"])

# Control deslizante para edad del cliente
age = st.sidebar.slider("Edad (Age)", 18, 92, 42)
balance = st.sidebar.number_input(  # Entrada numérica para saldo bancario
    "Saldo en Cuenta (USD)", 0.0, 250000.0, 85000.0, step=5000.0
)
# Control para historial crediticio
credit_score = st.sidebar.slider("Puntaje Crediticio", 350, 850, 650)

# Agrupa antigüedad y número de productos contratados
c_s3, c_s4 = st.sidebar.columns(2)
tenure = c_s3.number_input("Antigüedad (años)", 0, 10, 5)
num_products = c_s4.selectbox("Productos", [1, 2, 3, 4], index=0)

est_salary = st.sidebar.number_input(  # Entrada numérica para salario estimado anual
    "Salario Estimado (USD)", 10000.0, 200000.0, 100000.0, step=5000.0
)

# Casillas binarias de tarjeta de crédito y actividad
c_s5, c_s6 = st.sidebar.columns(2)
has_cr_card = 1 if c_s5.checkbox("Tarjeta Crédito", value=True) else 0
is_active = 1 if c_s6.checkbox("Miembro Activo", value=True) else 0

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
# Ordena las columnas exactamente como en entrenamiento
input_df = pd.DataFrame([input_dict])[feature_cols]

# Predicción real
# Probabilidad de abandono en porcentaje
churn_prob = model.predict_proba(input_df)[0][1] * 100
churn_pred = model.predict(input_df)[0]  # Clasificación final (0 o 1)

# 4. Layout en 2 columnas sin scroll
left_col, right_col = st.columns([1.08, 1], gap="medium")

with left_col:
    # Subtítulo de la tabla comparativa
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
    st.dataframe(  # Tabla interactiva nativa
        benchmark_df, use_container_width=True, hide_index=True, height=142
    )

    # Subtítulo de muestra real
    st.subheader(f"📋 Clientes Similares en Churn.csv ({geography})")
    filtered_df = raw_df[  # Filtra clientes reales del mismo país y rango de edad (±5 años)
        (raw_df["Geography"] == geography)
        & (raw_df["Age"] >= max(18, age - 5))
        & (raw_df["Age"] <= min(92, age + 5))
    ][[
        "Geography",
        "Gender",
        "Age",
        "Balance",
        "NumOfProducts",
        "IsActiveMember",
        "Exited",
    ]]

    st.dataframe(  # Muestra 6 casos reales similares para contexto operativo
        filtered_df.head(6), use_container_width=True, hide_index=True, height=225
    )

with right_col:
    # Subtítulo de KPIs en vivo
    st.subheader("🎯 Predicción en Tiempo Real del Cliente")

    m1, m2, m3 = st.columns(3)  # Fila de 3 tarjetas KPI nativas
    m1.metric("Probabilidad de Fuga", f"{churn_prob:.1f}%")
    m2.metric(
        "Diagnóstico",
        "Abandona (1)" if churn_pred == 1 else "Permanece (0)",
        delta="Riesgo Alto" if churn_pred == 1 else "Cliente Leal",
        delta_color="inverse" if churn_pred == 1 else "normal",
    )
    m3.metric("AUC-ROC Modelo", "0.8657", delta="F1: 0.6257 > 0.59")

    fig = px.bar(  # Gráfico de barras horizontales con las 8 variables más influyentes
        feature_imp.tail(8),
        x="Importancia",
        y="Variable",
        orientation="h",
        title="Top 8 Variables Más Influyentes en el Modelo",
    )
    fig.update_layout(  # Mantiene altura compacta para encajar en una sola pantalla
        height=285,
        margin=dict(l=10, r=10, t=35, b=10),
        xaxis_title="Peso relativo",
        yaxis_title="",
    )
    # Renderiza la gráfica responsiva
    st.plotly_chart(fig, use_container_width=True)
