#  Oil & Gas Predictive Maintenance using Machine Learning

An end-to-end Machine Learning project that predicts industrial equipment failures using sensor data. The project covers the complete ML lifecycle, including data preprocessing, exploratory data analysis (EDA), feature engineering, model building, evaluation, and deployment using Streamlit.

---

##  Project Objective

Unexpected equipment failures in the Oil & Gas industry can lead to production downtime, increased maintenance costs, and safety risks.

The objective of this project is to build a predictive maintenance system that identifies equipment failures in advance, enabling proactive maintenance and improving operational efficiency.

---

##  Project Workflow

- Data Collection
- Data Cleaning
- Handling Missing Values
- Exploratory Data Analysis (EDA)
- Feature Engineering
- One-Hot Encoding
- Train-Test Split
- Feature Scaling
- Model Training
- Model Evaluation
- Model Comparison
- Model Deployment using Streamlit

---

##  Dataset Information

- Total Records: **5,025**
- Industry: **Oil & Gas**
- Target Variable:
  - **0** → No Failure
  - **1** → Failure

### Features

- Temperature (°C)
- Pressure (bar)
- Vibration (mm/s)
- Flow Rate (m³/h)
- Motor Current (A)
- Oil Level (%)
- Runtime Hours
- Ambient Temperature (°C)
- Humidity (%)
- Maintenance Days
- Previous Failures
- Equipment Type

---

##  Machine Learning Models

The following models were trained and evaluated:

- Logistic Regression
- Logistic Regression (Balanced)
- Random Forest
- XGBoost

---

##  Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|------|---------:|----------:|--------:|---------:|---------:|
| Logistic Regression | 94.03% | 40.00% | 3.39% | 6.25% | 51.50% |
| **Logistic Regression (Balanced)**   | **78.01%** | 20.00% | **91.53%** | 32.83% | **84.00%** |
| Random Forest | 91.34% | 31.58% | 40.68% | **35.56%** | 67.59% |
| XGBoost | 94.03% | **46.15%** | 10.17% | 16.67% | 54.71% |

---

##  Final Model

**Selected Model:** Logistic Regression (Balanced)

### Reason?

Although Random Forest and XGBoost achieved higher overall accuracy, the balanced Logistic Regression model achieved the highest Recall (91.53%) and the best ROC-AUC (84.00%).

Since predictive maintenance focuses on identifying equipment failures before they occur, Recall was prioritized over Accuracy during model selection.

---

##  Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- XGBoost
- Streamlit
- Joblib

---

##  Streamlit Application

The application allows users to:

- Enter equipment sensor values
- Select equipment type
- Predict equipment failure
- View failure probability
- Identify risk level
- Receive maintenance recommendations

---

##  Project Structure

```
Predictive_Maintenance_Project/
│
├── app.py
├── predictive_maintenance_model.pkl
├── scaler.pkl
├── Oil_Gas_Predictive_Maintenance.ipynb
├── Oil_Gas_Predictive_Maintenance.xlsx
├── requirements.txt
└── README.md


##  Project links:
GitHub Repository: https://github.com/ayushmanral10-crypto/Oil-Gas-Predictive-Maintenance](https://github.com/ayushmanral10-crypto/Oil-Gas-Predictive-Maintenance.git
Live Streamlit App Link: https://oil-gas-predictive-maintenance-sq7rg5cfgp6yogczhfwgvx.streamlit.app/

---

##  Project Screenshots


![Uploading image.png…]()

> - EDA Visualizations
> - Correlation Heatmap
> - Model Comparison
> - Streamlit Dashboard
> - Prediction Result

---

##  Future Improvements

- Integrate real-time IoT sensor data
- Add cloud deployment
- Implement automated maintenance alerts
- Improve model performance with larger real-world datasets

---

##  Author

**Ayush Manral**
