# House Rent Prediction (Regression)

This project predicts monthly house rent prices using machine learning regression techniques.
The objective is to estimate rental costs based on property characteristics and location-related features.

---

## Project Overview

- **Problem Type:** Regression  
- **Target Variable:** `Rent`  
- **Dataset:** House Rent Dataset (public, CSV)  
- **Final Model:** Random Forest Regressor  
- **Deployment:** Streamlit app deployed on Hugging Face Spaces  

---

## Dataset Description

The dataset contains rental listings with both numerical and categorical features.

### Features
- `BHK` – number of bedrooms  
- `Size` – house size in square feet  
- `Bathroom` – number of bathrooms  
- `City` – city location  
- `Furnishing Status` – furnishing type  
- `Tenant Preferred` – tenant category  
- `Area Type` – property area type  

### Target
- `Rent` – monthly house rent  

---

## Project Structure

house_rent_prediction/
├── app.py
├── requirements.txt
├── models/
│ ├── house_rent_model.joblib
│ └── training_columns.json
├── notebooks/
│ └── house_rent_prediction.ipynb
└── README.md

---


---

## Methodology

### Exploratory Data Analysis (EDA)
- Analysis of rent distribution revealed strong right skewness  
- City and house size showed significant influence on rental prices  

### Preprocessing
- Numerical features scaled using `StandardScaler`  
- Categorical features encoded using `OneHotEncoder`  
- All preprocessing steps implemented via `sklearn` Pipelines  

### Target Transformation
- The target variable (`Rent`) was log-transformed using `log1p`  
- This reduced skewness and improved model performance  

### Modeling
- **Baseline Model:** Linear Regression  
- **Final Model:** Random Forest Regressor  
- Random Forest captured non-linear relationships more effectively  

### Evaluation Metrics
- Mean Absolute Error (MAE)  
- Root Mean Squared Error (RMSE)  
- R² Score (evaluated on log scale)  

---

## Model Performance Summary

- Linear Regression provides a basic baseline  
- Random Forest significantly reduces prediction error  
- Final model achieves strong predictive performance on a challenging dataset  

---

Conclusion

This project demonstrates a complete end-to-end regression workflow,
from data exploration and preprocessing to model training, evaluation, and deployment.

The use of log transformation combined with a Random Forest model significantly improves prediction accuracy, making the solution suitable for real-world rental price estimation tasks.

---
Future Improvements

Hyperparameter tuning

Additional feature engineering

Model comparison with Gradient Boosting and XGBoost

---


## How to Run Locally

```bash
git clone https://github.com/enesbayraktar61/house-rent-prediction.git
cd house-rent-prediction
pip install -r requirements.txt
streamlit run app.py
---
