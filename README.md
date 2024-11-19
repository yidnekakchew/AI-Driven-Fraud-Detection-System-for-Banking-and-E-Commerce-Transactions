# Fraud Detection System: Analysis, ML Models & Deployment
This project aims to build a comprehensive fraud detection and analysis system. It includes data preprocessing, model training and explainability, API development, and interactive dashboard visualization. Below is a breakdown of each task and its deliverables.

## Overview
* Advanced fraud detection system combining:
  - Data analysis
  - Machine learning
  - Model explainability
  - Interactive visualization

## Project Structure
```
Enhanced-Fraud-Detection-for-E-commerce-and-Banking-Transactions/
├── app/+
│ ├── README.md
│ ├── init.py
│ ├── app.py
│ ├── dashboard.py
│ └── templates/
├── flask_model_api/
│ ├── .dockerignore
│ ├── Dockerfile
│ ├── app.py
│ └── requirements.txt
├── notebooks/
│ ├── credit_card_preprocessing.ipynb
│ ├── feature_engineering.ipynb
│ ├── fraud_data_preprocessing.ipynb
│ ├── ipAddress_preprocessing.ipynb
│ ├── shap_lime_analysis.ipynb
│ └── tradition_model_training.ipynb
├── scripts/
│ ├── README.md
│ ├── init.py
│ ├── data_preprocessing.py
│ ├── feature_engineering.py
│ ├── merge_dataset.py
│ ├── shap_lime_analysis.py
│ └── traditional_model_training.py
├── tests/
├── .gitignore
└── requirements.txt
└── README.md
```

## Key Features
* Data Processing Pipeline (**Task1 branch**)
  - Missing value handling
  - Data cleaning & transformation
  - Feature engineering
  - Geolocation analysis

* Machine Learning Models (**Task2 branch**)
  - Traditional ML algorithms
  - Deep Learning models
  - MLflow integration

* Model Explainability
  - SHAP implementation
  - LIME integration
  - Feature visualization

* Deployment
  - REST API
  - Docker support
  - System logging

* Interactive Dashboard
  - Real-time monitoring
  - Transaction analysis
  - Device statistics

## Technical Implementation

### Data Analysis & Preprocessing
* Missing value handling
* Duplicate removal
* Data type corrections
* Feature engineering:
  - Transaction metrics
  - Time features
  - IP processing

### Model Building
* Implemented algorithms:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
  - Neural Networks
* MLflow experiment tracking

# Clone repository
git clone https://github.com/Endekalu777/Enhanced-Fraud-Detection-for-E-commerce-and-Banking-Transactions.git

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Unix
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Usage
API: Access the model API endpoints to make predictions and view model outputs.
Dashboard: Open the dashboard in your browser to view visualizations and insights about the fraud data.

# License
This project is licensed under the MIT License - see the LICENSE file for details.

# Contact
For any questions or additional information, please contact [Endekalu.simon@gmail.com](mailto:Endekalu.simon@gmail.com)