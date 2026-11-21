[🇺🇸 Read in English](#-beta-bank--customer-churn-prediction) | [🇲🇽 Leer en Español](#-beta-bank--predicción-de-abandono-de-clientes-churn)

---

# 🏦 Beta Bank — Customer Churn Prediction

> Machine Learning classification model to anticipate customer attrition, implementing advanced class-balancing techniques and hyperparameter tuning to exceed the required quality threshold (F1-score > 0.59).

🚀 **[Launch Interactive Streamlit App](https://beta-bank-churn-prediction.streamlit.app/)** | 🌐 **[View Quarto Web Report](https://andagui-ds.github.io/beta-bank-churn-prediction/)**

---

## 🎯 Business Problem

**Beta Bank** customers are leaving the institution month by month. Retaining an existing customer is far more cost-effective than acquiring a new one, so the bank needs to proactively identify users with a high probability of closing their accounts.

The main challenge lies in the **severe class imbalance**: approximately 80% of customers stay (Class 0) and only 20% leave (Class 1). A traditional model would tend to always predict "stays," failing to identify churners.

## 📊 Data

* **Source:** `Churn.csv`
* **Records:** 10,000 customers.
* **Predictor Features:** Credit score, geography, gender, age, tenure (`Tenure`), account balance, number of products, credit card ownership, active membership, and estimated salary.
* **Target Variable:** `Exited` (1 = Left the bank, 0 = Stayed).

## 🧠 Approach and Solution

1. **Preprocessing:**
   * Dropped non-predictive columns (`RowNumber`, `CustomerId`, `Surname`).
   * Imputed missing values in `Tenure` using the median.
   * Applied *One-Hot Encoding* for categorical features.
   * Scaled numerical features with `StandardScaler`.
2. **Data Splitting:** Train (60%), Validation (20%), and Test (20%), applying stratification to preserve the 80/20 ratio across all three sets.
3. **Baseline Model:** Unbalanced Random Forest to establish a benchmark (F1 ≈ 0.55).
4. **Imbalance Handling (Two approaches evaluated):**
   * **Class Weighting:** `class_weight='balanced'` (F1 ≈ 0.55).
   * **Upsampling:** Multiplied minority class observations (churners) by 4 in the training set.
5. **Optimization:** Hyperparameter search (`n_estimators` and `max_depth`) for the Random Forest trained on upsampled data, evaluated on the validation set.

## 🏆 Results and Performance

The **Upsampling + Hyperparameter Tuning** approach proved to be the most effective:

| Model | F1-Score (Validation) | F1-Score (Test) | AUC-ROC (Test) |
| :--- | :---: | :---: | :---: |
| **Random Forest (Upsampling + Tuning)** | **0.6406** | **0.6257** | **0.8657** |
| Random Forest (Class Weight) | 0.5522 | - | - |
| Random Forest (Baseline) | 0.5556 | - | - |
| Dummy (Majority Class) | 0.00 | - | 0.50 |

* **F1-Score:** The final model achieved **0.6257**, successfully surpassing the minimum required threshold of 0.59.
* **AUC-ROC:** A score of **0.8657** demonstrates strong discriminative power; the model distinguishes a churner from a loyal customer far better than a random classifier.

## ✅ Business Recommendation

The upsampled Random Forest model is ready for production. Beta Bank can use these predictions to launch targeted retention campaigns (special offers, loyalty calls) exclusively for users flagged as likely to churn, optimizing the marketing budget and reducing the overall churn rate.

## 🛠️ Tech Stack

* **Language:** Python
* **Libraries:** Pandas, Scikit-learn, Matplotlib, Seaborn, Plotly, Streamlit
* **Techniques:** Random Forest, Upsampling, Classification, Hyperparameter Tuning, ROC Curve.

## 🔗 Full Analysis & Production App

* 📓 [Original Jupyter Notebook with full step-by-step code](notebooks/01-analisis-churn.ipynb)
* 🚀 [Interactive Customer Churn Predictor (Streamlit)](https://beta-bank-churn-prediction.streamlit.app/)

## ▶️ How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
