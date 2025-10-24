# House Price Analysis Project 🏠

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-orange)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-Educational-green)](LICENSE)

This repository contains a complete workflow for analyzing, predicting, and deploying a **house price prediction model** using machine learning and a Python web app interface.

---

## Project Structure

house_prices_regression/
│
├── data/
│   ├── real_estate_dataset.csv           # Original dataset from Kaggle
│   ├── real_estate_dataset copy.csv      # Copy of the original dataset (good practice)
│   └── cleaned_realEstate.csv            # Cleaned dataset (processed using Excel)
│
├── notebooks/
│   └── house_prices_analysis.ipynb       # Jupyter Notebook with:
│       - Exploratory Data Analysis (EDA)
│       - Model training and evaluation
│       - Model deployment code
│
├── housePricesApp/
│   ├── app.py                            # Streamlit web app to launch the model
│   ├── requirements.txt                  # Python dependencies for Streamlit app
│   └── house_prices_model.pkl            # Serialized trained ML model
│
└── README.md                             # Project documentation


---

## Description

This project performs **house price analysis** using a dataset of real estate properties. The workflow includes:

1. **Data Cleaning and Preprocessing:** Handling missing values and ensuring data consistency.
2. **Exploratory Data Analysis (EDA):** Understanding the distribution and relationships of features.
3. **Machine Learning Models:**
   - Linear Regression
   - Random Forest Regressor
   - XGBoost Regressor (including hyperparameter tuning)
4. **Model Evaluation:** Metrics used:
   - Mean Absolute Error (MAE)
   - Mean Squared Error (MSE)
   - Root Mean Squared Error (RMSE)
   - Root Mean Squared Log Error (RMSLE)
   - R² Score
   - Approximate accuracy within a custom tolerance
5. **Model Deployment:** A Streamlit app that allows users to input house features and get a predicted price.

---

## Usage

### 1. Google colab Notebook
- Open `notebooks/house_prices_analysis.ipynb` in colab.
- Run cells sequentially to perform EDA, train models, and evaluate predictions.

### 2. Streamlit Web App
1. Navigate to the `housePricesApp` folder.
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
3. Run the Streamlit app:
    streamlit run app.py
4. Open the browser interface, enter house details, and click Predict Price to see predictions.

**Dataset**
- Original dataset: data/real_estate_dataset.csv (downloaded from Kaggle)
- Working dataset: data/cleaned_realEstate.csv (cleaned and processed version)
- Always use a copy of the original dataset for reproducibility.

**Tableau Visualizations**
- Workbook: realEstateWorkbook.twb
- Includes interactive visualizations of house prices and feature analysis.

**Model**
- The trained model is stored as a .pkl file (house_prices_model.pkl) and loaded in the Streamlit app for real-time predictions.
- Models used: Linear Regression, Random Forest, XGBoost (tuned).

**Dependencies**
- pandas
- numpy
- scikit-learn
- xgboost
- streamlit
- joblib
- matplotlib / seaborn (for visualizations in the notebook)

**Author**
Koller Melanie Turinabo
Third-year Computer Science & Systems Engineering student
[kollermelaniet@gmail.com] | (https://www.linkedin.com/in/koller-melanie-turinabo-963289309/)

**License**
This project is for educational purposes. Please do not redistribute without permission.

