# 🚑 Random_Forest Model For Predicting Diabites 🚑

open demo app either clicking on link or copy the url.

Website streamlit app link: [Random_Forest Model App For Predicting Diabites](https://random-forest-for-diabites.streamlit.app/)

or

copy this url: https://random-forest-for-diabites.streamlit.app/




## 📌 Project Overview

This notebook focuses on preparing raw diabetes prediction data for machine learning models. The process includes checking data distributions, handling missing values, deduplicating records, and encoding categorical variables.

---

## 🛠️ Data Preprocessing Workflow

### 1. Data Loading & Initial Inspection
* **Dataset:** `1000rows_diabetes_prediction_small.csv` loaded into Pandas via Google Drive.
* **Initial Shape:** 1,000 rows × 9 columns.
* **Features:**
  * **Categorical / Binary:** `gender`, `hypertension`, `heart_disease`, `smoking_history`, `diabetes`
  * **Numerical:** `age`, `bmi`, `HbA1c_level`, `blood_glucose_level`

### 2. Feature & Value Analysis
* **Unique Values Check:** Verified distributions across all features (`gender`, `smoking_history`, target `diabetes`, etc.).
* **Categorical Breakdown:** Identified 6 distinct categories in `smoking_history` (`'No Info'`, `'former'`, `'never'`, `'not current'`, `'current'`, `'ever'`).
* **Value Counts:** Ran frequency inspections to detect anomalies or extreme class imbalances.

### 3. Data Integrity & Missing Values
* **Null Check:** Checked across all columns using `.isna().sum()`.
* **Result:** **0 missing values** detected across all features.

### 4. Data Cleaning & Feature Transformation
To preserve the raw data, modifications were performed on a working copy (`df1`):

* **Deduplication:**
  * Ran `df1.drop_duplicates()`
  * **Removed:** 2 duplicate rows (reducing dataset from 1,000 to **998 rows**).
* **Categorical Encoding:**
  * One-hot encoded `gender` using `pd.get_dummies(drop_first=True, dtype=int)`.
  * Transformed `gender` into a binary numerical column `gender_Male` (`1` = Male, `0` = Female).

