import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.simplefilter('ignore')
import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, precision_recall_curve, auc)
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from imblearn.over_sampling import SMOTE
import pickle

# Set MLflow tracking URI to store in Google Drive
base_save_dir = "/content/drive/MyDrive/credit_fraud_model_training"
mlflow_tracking_dir = os.path.join(base_save_dir, "mlruns")  # Create specific directory for MLflow
os.makedirs(mlflow_tracking_dir, exist_ok=True)
mlflow.set_tracking_uri(f"file://{mlflow_tracking_dir}")

# Create or get experiment
experiment_name = "fraud_detection"
try:
    existing_exp = mlflow.get_experiment_by_name(experiment_name)
    if existing_exp:
        experiment_id = existing_exp.experiment_id
    else:
        experiment_id = mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)
except Exception as e:
    print(f"Error setting up MLflow experiment: {e}")
    raise

# logging path
logging.basicConfig(filename=os.path.join(base_save_dir, 'model_training.log'),
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filemode='a')

class ModelTraining:
    def __init__(self, file_path1, file_path2):
        print(f"Initializing ModelTraining with datasets {file_path1} and {file_path2}")
        logging.info(f"Initializing ModelTraining with datasets {file_path1} and {file_path2}")
        
        try:
            self.credit_df = pd.read_csv(file_path1)
            self.fraud_df = pd.read_csv(file_path2)

            self.fraud_df = pd.get_dummies(self.fraud_df, columns=['source'], drop_first=True)
            self.fraud_df = self.fraud_df.drop(columns=['device_id', 'user_id'], axis=1)
            logging.info("Successfully loaded and processed datasets.")
        except FileNotFoundError as e:
            logging.error(f"File not found: {e}")
            raise
        except pd.errors.EmptyDataError as e:
            logging.error(f"Empty data error: {e}")
            raise
        except Exception as e:
            logging.error(f"Error loading datasets: {e}")
            raise

    def data_preparation(self):
        try:
            logging.info("Started data preparation.")
            
            # Credit Card dataset preparation
            self.X_creditcard = self.credit_df.drop(columns=['Class'])
            self.y_creditcard = self.credit_df['Class']
            
            # Train-test split for credit card data
            self.X_train_cc, self.X_test_cc, self.y_train_cc, self.y_test_cc = train_test_split(
                self.X_creditcard, self.y_creditcard, test_size=0.3, random_state=42
            )
            
            logging.info("Separated features and target for Credit Card dataset.")

            # Fraud dataset preparation
            if 'signup_time' in self.fraud_df.columns and 'purchase_time' in self.fraud_df.columns:
                self.fraud_df['signup_time'] = pd.to_datetime(self.fraud_df['signup_time'])
                self.fraud_df['purchase_time'] = pd.to_datetime(self.fraud_df['purchase_time'])
                self.fraud_df['signup_time'] = self.fraud_df['signup_time'].apply(lambda x: x.timestamp())
                self.fraud_df['purchase_time'] = self.fraud_df['purchase_time'].apply(lambda x: x.timestamp())
                logging.info("Converted signup_time and purchase_time columns to timestamp.")

            self.X_fraud = self.fraud_df.drop(columns=['class'])
            self.y_fraud = self.fraud_df['class']

            # Train-test split for fraud data
            self.X_train_fraud, self.X_test_fraud, self.y_train_fraud, self.y_test_fraud = train_test_split(
                self.X_fraud, self.y_fraud, test_size=0.3, random_state=42
            )

            try:
                # Apply SMOTE to balance the fraud dataset
                smote = SMOTE(random_state=42)
                self.X_train_fraud, self.y_train_fraud = smote.fit_resample(self.X_train_fraud, self.y_train_fraud)
                logging.info("Applied SMOTE to oversample the minority class.")
            except Exception as e:
                logging.error(f"Error applying SMOTE: {e}")
                raise

            logging.info("Data preparation completed successfully.")
            
        except KeyError as e:
            logging.error(f"Missing columns in dataset: {e}")
            raise
        except Exception as e:
            logging.error(f"Error during data preparation: {e}")
            raise

    def evaluate_model(self, model, X_test, y_test, dataset_name):
        """Evaluate the model using various metrics and log the results."""
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall_vals, precision_vals)

        # Print metrics
        print(f"\n{dataset_name} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"PR-AUC: {pr_auc:.4f}\n")

        # Log metrics
        logging.info(f"\n{dataset_name} Results:")
        logging.info(f"Accuracy: {accuracy:.4f}")
        logging.info(f"Precision: {precision:.4f}")
        logging.info(f"Recall: {recall:.4f}")
        logging.info(f"F1-score: {f1:.4f}")
        logging.info(f"ROC-AUC: {roc_auc:.4f}")
        logging.info(f"PR-AUC: {pr_auc:.4f}\n")

        mlflow.log_metrics({
            f"{dataset_name}_accuracy": accuracy,
            f"{dataset_name}_precision": precision,
            f"{dataset_name}_recall": recall,
            f"{dataset_name}_f1": f1,
            f"{dataset_name}_roc_auc": roc_auc,
            f"{dataset_name}_pr_auc": pr_auc
        })

    def save_model(self, model, model_name, dataset_name):
        """
        Saves the trained model as a pickle file in the base directory.
        """
        try:
            pickle_path = os.path.join(base_save_dir, f"{model_name.lower()}_{dataset_name}_model.pkl")
            with open(pickle_path, 'wb') as f:
                pickle.dump(model, f)
            logging.info(f"Model saved as pickle file at {pickle_path}")
            return pickle_path
        except Exception as e:
            logging.error(f"Error saving model as pickle: {e}")
            raise

    def train_creditcard_models(self):
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(),
            'Random Forest': RandomForestClassifier()
        }

        param_grids = {
            'Logistic Regression': {
                'C': [0.001, 0.01, 0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            },
            'Decision Tree': {
                'max_depth': [3, 5, 7, 10],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'Random Forest': {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5]
            }
        }

        # Create parent run
        with mlflow.start_run(run_name="Credit Card Model Training - Parent") as parent_run:
            parent_run_id = parent_run.info.run_id

            for name, model in models.items():
                # Create nested run for each model
                with mlflow.start_run(run_name=f"Credit Card - {name}", nested=True) as child_run:
                    logging.info(f"Training {name} with GridSearchCV...")
                    print(f"Training {name} on Credit Card dataset...")
                    try:
                        grid_search = GridSearchCV(model, param_grids[name], cv=5, n_jobs=-1)
                        grid_search.fit(self.X_train_cc, self.y_train_cc)

                        best_model = grid_search.best_estimator_

                        # Create signature and input example
                        signature = infer_signature(self.X_train_cc, self.y_train_cc)
                        input_example = self.X_train_cc.iloc[:5]

                        self.evaluate_model(best_model, self.X_test_cc, self.y_test_cc, f"{name} on Credit Card Dataset")

                        # Log parameters and model in the child run
                        mlflow.log_param("Model Name", name)
                        mlflow.log_params(grid_search.best_params_)
                        
                        # Use a simple relative path for model saving
                        model_path = f"{name.replace(' ', '_').lower()}_model"
                        mlflow.sklearn.log_model(
                            best_model, 
                            model_path,
                            signature=signature,
                            input_example=input_example
                        )
                        logging.info(f"Saved {name} model to MLflow")

                        # Save model as a pickle file
                        pickle_path = self.save_model(best_model, name.replace(' ', '_'), 'creditcard')
                        mlflow.log_artifact(pickle_path, "pickle_files")
                        logging.info(f"Saved {name} model to MLflow and as pickle file")

                    except Exception as e:
                        logging.error(f"Error training {name}: {e}")
                        raise

    def train_fraud_models(self):
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, solver='lbfgs', class_weight='balanced'),
            'Decision Tree': DecisionTreeClassifier(class_weight='balanced'),
            'Random Forest': RandomForestClassifier(class_weight='balanced')
        }
        param_grids = {
            'Logistic Regression': {
                'C': [0.001, 0.01, 0.1, 1, 10],
                'penalty': ['l2'],
                'solver': ['lbfgs']
            },
            'Decision Tree': {
                'max_depth': [3, 5, 7, 10],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'Random Forest': {
                'n_estimators': [50, 100],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5]
            }
        }

        # Create parent run
        with mlflow.start_run(run_name="Fraud Model Training - Parent") as parent_run:
            for name, model in models.items():
                # Create nested run for each model
                with mlflow.start_run(run_name=f"Fraud - {name}", nested=True) as child_run:
                    logging.info(f"Training {name} with GridSearchCV...")
                    print(f"Training {name} on Fraud dataset...")
                    try:
                        grid_search = GridSearchCV(model, param_grids[name], cv=3, n_jobs=-1, scoring='f1')  # Optimize for F1-score
                        grid_search.fit(self.X_train_fraud, self.y_train_fraud)

                        best_model = grid_search.best_estimator_

                        # Create signature and input example
                        signature = infer_signature(self.X_train_fraud, self.y_train_fraud)
                        input_example = self.X_train_fraud.iloc[:5]

                        # Evaluate the best model
                        self.evaluate_model(best_model, self.X_test_fraud, self.y_test_fraud, f"{name} on Fraud Dataset")

                        # Log parameters and model in the child run
                        mlflow.log_param("Model Name", name)
                        mlflow.log_params(grid_search.best_params_)
                        
                        # Use a simple relative path for model saving
                        model_path = f"{name.replace(' ', '_').lower()}_fraud_model"
                        mlflow.sklearn.log_model(
                            best_model, 
                            model_path,
                            signature=signature,
                            input_example=input_example
                        )
                        logging.info(f"Saved {name} model to MLflow")
                        # Save pickle file
                        pickle_path = self.save_model(best_model, name.replace(' ', '_'), 'fraud')
                        mlflow.log_artifact(pickle_path, "pickle_files")
                        logging.info(f"Saved {name} model to MLflow and pickle file")
                

                    except Exception as e:
                        logging.error(f"Error training {name}: {e}")
                        raise