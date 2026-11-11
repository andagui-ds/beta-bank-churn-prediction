# 🏦 Beta Bank — Predicción de Abandono de Clientes (Churn)

> Modelo de clasificación de Machine Learning para anticipar la fuga de clientes, implementando técnicas avanzadas de balanceo de clases y optimización de hiperparámetros para superar el umbral de calidad exigido (F1-score > 0.59).

## 🎯 Problema de Negocio

Los clientes de **Beta Bank** están abandonando la institución mes a mes. Retener a un cliente existente es mucho más rentable que adquirir uno nuevo, por lo que el banco necesita identificar de manera proactiva a los usuarios con alta probabilidad de cancelar sus cuentas. 

El desafío principal radica en el **fuerte desequilibrio de clases**: aproximadamente el 80% de los clientes se quedan (Clase 0) y solo el 20% se van (Clase 1). Un modelo tradicional tendería a predecir siempre "se queda", fallando en la identificación de los desertores.

## 📊 Datos

- **Fuente:** `Churn.csv`
- **Registros:** 10,000 clientes.
- **Variables predictoras:** Puntuación crediticia, geografía, género, edad, antigüedad (`Tenure`), saldo, número de productos, posesión de tarjeta, membresía activa y salario estimado.
- **Variable objetivo:** `Exited` (1 = Abandonó el banco, 0 = Se quedó).

## 🧠 Enfoque y Solución

1. **Preprocesamiento:** 
   - Eliminación de columnas no predictivas (`RowNumber`, `CustomerId`, `Surname`).
   - Imputación de valores nulos en `Tenure` con la mediana.
   - Codificación *One-Hot Encoding* para variables categóricas.
   - Escalado de variables numéricas con `StandardScaler`.
2. **División de datos:** Entrenamiento (60%), Validación (20%) y Prueba (20%), aplicando estratificación para mantener la proporción 80/20 en los tres conjuntos.
3. **Modelo Base:** Random Forest sin balanceo, para establecer una línea base (F1 ≈ 0.55).
4. **Manejo del Desbalanceo (Dos enfoques evaluados):**
   - **Peso de clase:** `class_weight='balanced'` (F1 ≈ 0.55).
   - **Sobremuestreo (Upsampling):** Multiplicar las observaciones de la clase minoritaria (desertores) por 4 en el conjunto de entrenamiento.
5. **Optimización:** Búsqueda de hiperparámetros (`n_estimators` y `max_depth`) para el Random Forest entrenado con datos sobremuestreados, evaluado en el conjunto de validación.

## 🏆 Resultados y Rendimiento

El enfoque de **Sobremuestreo + Ajuste de Hiperparámetros** demostró ser el más efectivo.

| Modelo | F1-Score (Validación) | F1-Score (Prueba) | AUC-ROC (Prueba) |
| :--- | ---: | ---: | ---: |
| **Random Forest (Upsampling + Tuning)** | **0.6406** | **0.6257** | **0.8657** |
| Random Forest (Class Weight) | 0.5522 | - | - |
| Random Forest (Base) | 0.5556 | - | - |
| Dummy (Clase mayoritaria) | 0.00 | - | 0.50 |

- **F1-Score:** El modelo final alcanzó un **0.6257**, superando con éxito el umbral mínimo exigido de 0.59.
- **AUC-ROC:** Un valor de **0.8657** demuestra una excelente capacidad de discriminación; el modelo distingue a un desertor de un cliente leal mucho mejor que un clasificador aleatorio.

## ✅ Recomendación de Negocio

El modelo de Random Forest con sobremuestreo está listo para producción. Beta Bank puede utilizar estas predicciones para crear campañas de retención dirigidas (ofertas, llamadas de fidelización) únicamente a los usuarios que el modelo marque como propensos a abandonar, optimizando el presupuesto de marketing y reduciendo la tasa de churn.

## 🛠️ Stack Técnico
- **Lenguajes:** Python
- **Librerías:** Pandas, Scikit-learn, Matplotlib, Seaborn
- **Técnicas:** Random Forest, Upsampling, Clasificación, Optimización de hiperparámetros, Curva ROC.

## 🔗 Ver el Análisis Completo
- 📓 [Notebook original con todo el código paso a paso](notebooks/analisis-beta-bank.ipynb)
## ▶️ Cómo ejecutar

```bash
pip install -r requirements.txt
jupyter notebook notebooks/01-analisis-churn.ipynb
