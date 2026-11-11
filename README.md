# 🏦 Reducir el churn bancario: por qué accuracy miente y F1 no

**Problema de negocio:** Beta Bank está perdiendo clientes mes a mes. El objetivo es construir un modelo predictivo capaz de identificar a los clientes con alta probabilidad de abandonar el banco, para priorizar campañas de retención. 

**El reto técnico:** El dataset está desbalanceado (aprox. 80% se queda, 20% se va). Por ello, la métrica de Accuracy es engañosa. El modelo fue optimizado para alcanzar un **F1-score mínimo de 0.59**, equilibrando Precision y Recall.

---

## 📊 Resultado principal

- **F1-score final:** **0.6257** (superando el umbral exigido de 0.59) ✅
- **AUC-ROC final:** **0.8657** (alta capacidad de discriminación)
- **Técnica ganadora:** Sobremuestreo (Upsampling) de la clase minoritaria + Random Forest optimizado.

---

## 🛠️ Metodología

1. **Limpieza:** Imputación de nulos en `Tenure` (mediana), eliminación de columnas irrelevantes (`RowNumber`, `CustomerId`, `Surname`) y One-Hot Encoding para variables categóricas.
2. **Preprocesamiento:** Estandarización de variables numéricas (`StandardScaler`) ajustado solo con el set de entrenamiento para evitar fuga de datos (data leakage).
3. **Manejo del desbalance:** Se probaron dos enfoques: `class_weight='balanced'` y UpSampling de la clase 1. El UpSampling demostró mejor rendimiento.
4. **Modelado:** Búsqueda de hiperparámetros (`n_estimators` y `max_depth`) para un clasificador Random Forest.
5. **Evaluación:** Uso de validación cruzada (5-fold) para robustez y evaluación final en un set de prueba aislado (20%).

---

## 📁 Contenido del repositorio

- `notebooks/01-analisis-churn.ipynb`: Análisis completo, balanceo, modelado y matriz de confusión.
- `requirements.txt`: Dependencias del proyecto.
- `models/`: Almacena el modelo entrenado (`.joblib`) listo para producción.

---

## ▶️ Cómo ejecutar

```bash
pip install -r requirements.txt
jupyter notebook notebooks/01-analisis-churn.ipynb