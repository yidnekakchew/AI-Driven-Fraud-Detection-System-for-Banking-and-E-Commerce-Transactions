import pickle
import pandas as pd
import numpy as np
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (17, 7)  

# Load your models
model_names = [
    "../models/random_forest_fraud_model.pkl",
    "../models/decision_tree_fraud_model.pkl",
    "../models/random_forest_creditcard_model.pkl",
    "../models/decision_tree_creditcard_model.pkl"
]

models = {}
for model_name in model_names:
    with open(model_name, 'rb') as file:
        models[model_name] = pickle.load(file)

# Load your datasets
fraud_data = pd.read_csv("../data/fraud_model_training.csv")
creditcard_data = pd.read_csv("../data/credit_model_training.csv")

# Prepare feature matrices
X_fraud = fraud_data.drop(columns=["class"])  
X_creditcard = creditcard_data.drop(columns=["Class"])


# Function for SHAP analysis
def shap_analysis(model, X, model_name):
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)

    # Calculate SHAP values
    shap_values = explainer.shap_values(X)

    # Summary Plot
    print(f"SHAP Summary Plot for {model_name}")
    plt.figure(figsize=(10, 6))  # Set figure size
    if isinstance(shap_values, list):
        shap.summary_plot(shap_values[1], X)
    else:
        shap.summary_plot(shap_values, X)
    plt.tight_layout()  # Adjust layout
    plt.show()

    # Force Plot for the first instance
    print(f"SHAP Force Plot for {model_name} (first instance)")
    try:
        plt.figure(figsize=(10, 4))  # Set figure size
        if isinstance(shap_values, list):
            single_instance = X.iloc[[0]]  # Keep as DataFrame with one row
            instance_values = shap_values[1][0]  # First row values
            shap.plots.force(
                explainer.expected_value[1],
                instance_values,
                features=single_instance.iloc[0],
                feature_names=X.columns.tolist()
            )
        else:
            single_instance = X.iloc[[0]]
            instance_values = shap_values[0]
            shap.plots.force(
                explainer.expected_value,
                instance_values,
                features=single_instance.iloc[0],
                feature_names=X.columns.tolist()
            )
    except Exception as e:
        print("Error generating force plot:", e)

    plt.tight_layout()  # Adjust layout
    plt.show()


def lime_analysis(model, X, model_name):
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X),
        feature_names=X.columns,
        class_names=['Class 0', 'Class 1'],  # Update with your class names
        mode='classification'
    )

    # Explain a specific prediction (first instance)
    instance_to_explain = X.iloc[0]
    exp = lime_explainer.explain_instance(
        data_row=instance_to_explain,
        predict_fn=model.predict_proba
    )
    
    # Feature Importance Plot
    print(f"LIME Feature Importance Plot for {model_name} (first instance)")
    exp.as_pyplot_figure()
    plt.title(f'LIME Feature Importance for {model_name}')
    plt.show()



# Function to run all analyses
def run_analyses():
    print("Analyzing Fraud Detection Models...")
    for model_name, model in models.items():
        if 'fraud' in model_name.lower():
            print(f"\nAnalyzing {model_name}")
            shap_analysis(model, X_fraud, f"Fraud Model - {model_name}")
            lime_analysis(model, X_fraud, f"Fraud Model - {model_name}")
            
    print("\nAnalyzing Credit Card Models...")
    for model_name, model in models.items():
        if 'creditcard' in model_name.lower():
            print(f"\nAnalyzing {model_name}")
            shap_analysis(model, X_creditcard, f"Credit Card Model - {model_name}")
            lime_analysis(model, X_creditcard, f"Credit Card Model - {model_name}")

