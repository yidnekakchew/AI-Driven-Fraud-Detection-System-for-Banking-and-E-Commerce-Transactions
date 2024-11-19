import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from IPython.display import display
import warnings
from datetime import datetime
import os


# Suppress specific warnings(FutureWarnings and UserWarnings) to reduce unnecessary output during analysis.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Get the current working directory
current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)

# Ensure the logging directory exists
log_directory = os.path.join(parent_directory, 'logging')
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

# Create a timestamp for the log file name
def get_log_file_name(filepath):
    dataset_name = os.path.splitext(os.path.basename(filepath))[0]  # Extract dataset name from filepath
    log_file_name = f'data_preprocessing_{dataset_name}.log'
    log_file_path = os.path.join(log_directory, log_file_name)
    return log_file_path

class pre_processing():
    """
    Class for preprocessing and analyzing datasets.
    Includes methods for loading data, cleaning, univariate/bivariate analysis, 
    and logging for easy tracking of operations.
    """
    def __init__(self, filepath):
        self.filepath = filepath
        log_file_path = get_log_file_name(filepath)
        # Configure logging to append if the file exists
        logging.basicConfig(filename=log_file_path,
                            level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            filemode='a')  # Always append to the existing log file

        logging.info(f"Log file opened at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} for dataset {filepath}")

        # Load the CSV file into a DataFrame using pandas
        try:
            self.df = pd.read_csv(filepath)
            logging.info("Data successfully loaded from: %s", filepath)
        except Exception as e:
            logging.error(f"Error loading data from {filepath}: {e}")
            raise

    def data_overview(self):
        """
        Provide an overview of the dataset structure, including shape, preview, 
        data types, and missing values.
        """
        # Shape of the data
        try:
            logging.info("Generating data overview")
            print(f"Shape of the dataset: {self.df.shape}\n")
            logging.info("Successfully retrieved the shape of the dataset.")
        except Exception as e:
            logging.error(f"Error retrieving the shape of the dataset: {e}")

        # Used head to preview first 5 rows
        try:
            logging.info("Displaying the first five rows of the data")
            print("Preview of the first 5 rows:")
            display(self.df.head())
            logging.info("Successfully displayed the first five rows.")
        except Exception as e:
            logging.error(f"Error displaying the first five rows.")

        # Showing the data types in the dataset
        try:
            logging.info("Displaying data types of the dataset")
            print("Data types of the dataset")
            data_types = self.df.dtypes
            display(data_types)
            logging.info("Successfully displayed the data types of the dataset")
        except Exception as e:
            logging.error(f"Error displaying the data types of the dataset {e}")
    
        # Overview of missing values of the dataset
        try:
            print("Missing values of the dataset: ")
            missing_values_count = self.df.isnull().sum() # count missing values 
            total_entries = self.df.shape[0] # Total number of entries
            missing_values_percentage = (missing_values_count / total_entries) * 100 # Calculate percentage

            # create dataframe for missing values
            missing_values_df = pd.DataFrame({
                'Missing Values Count': missing_values_count,
                'Missing Values Percentage (%)': missing_values_percentage
            })
            print("Missing values in the dataset (Count and percentage)")
            display(missing_values_df)
            logging.info("Successfully displayed the missing values")
        except Exception as e:
            logging.error(f"Error displaying the missing values of the dataset: {e}")


    def data_cleaning(self):
        """
        Cleans the dataset by removing duplicates and correcting data types for specific columns
        such as 'signup_time', 'purchase_time', and IP addresses.
        Logs every step of the process for traceability.
        """
        logging.info("Starting data cleaning process")
        try:
            # Remove dupliates
            self.df.drop_duplicates(inplace = True)
            logging.info("Completed removing duplicates")

            # Correct data types
            if 'signup_time' in self.df.columns:
                self.df['signup_time'] = pd.to_datetime(self.df['signup_time'])
                logging.info("Converted 'signup_time' to datetime")

            if 'purchase_time' in self.df.columns:
                self.df['purchase_time'] = pd.to_datetime(self.df['purchase_time'])
                logging.info("Converted 'purchase_time' to datetime.")

            for column in ['lower_bound_ip_address', 'upper_bound_ip_address', 'ip_address']:
                if column in self.df.columns:
                        # Ensure the column is a float before conversion and handle missing values
                        if self.df[column].dtype == 'float64':
                            # Convert IP addresses from float to integer, handling NaNs appropriately
                            self.df[column] = self.df[column].fillna(0).astype(int)
                            logging.info(f"Converted '{column}' from float to integer, with missing values handled.")
                        else:
                            logging.warning(f"'{column}' is not of type float64, skipping conversion.")
        except Exception as e:
            logging.info(f"Error during data cleaning: {e}")

    def get_suitable_columns(self):
        """
        Identify which columns are suitable for univariate and bivariate analysis 
        based on the specific dataset.
        """
        dataset_name = os.path.basename(self.filepath).lower()  # Access the filepath directly from the class attribute

        # Define columns that are not suitable for univariate or bivariate analysis
        unsuitable_columns = []

        # Check the dataset name and assign unsuitable columns accordingly
        if 'credit' in dataset_name.lower():
            unsuitable_columns = ['Time'] + [f'V{i}' for i in range(1, 29)]
        elif 'ip' in dataset_name.lower():
            unsuitable_columns = ['lower_bound_ip_address', 'upper_bound_ip_address']
        elif 'fraud' in dataset_name.lower():
            unsuitable_columns = ['user_id', 'device_id', 'ip_address', 'signup_time', 'purchase_time']

        # Suitable columns for analysis are those not in the unsuitable list
        suitable_columns = [col for col in self.df.columns if col not in unsuitable_columns]
        
        logging.info(f"Identified suitable columns for analysis in {dataset_name}: {suitable_columns} for univariate and bivariate analysis")
        return suitable_columns
    
    def univariate_analysis(self):
        """
        Perform univariate analysis specifically for 'Amount' and 'Class' columns 
        in any dataset where they exist. Optimize for efficiency without redundant checks.
        """
        logging.info("Starting univariate analysis")

        # Check if 'Amount' exists in the dataset for a histogram plot
        if 'Amount' in self.df.columns:
            try:
                logging.info("Performing histplot for 'Amount' column")
                print("\nUnivariate Analysis for: Amount")

                # Plot the distribution of 'Amount' with optimized bins for better visibility
                plt.figure(figsize=(10, 6))
                sns.histplot(self.df['Amount'], bins=50, kde=True)  # Adjust bins as needed
                plt.title("Distribution of Amount")
                plt.xlabel('Amount')
                plt.ylabel('Frequency')
                plt.show()  # Ensure the plot is displayed

                # Display summary statistics for 'Amount'
                print(self.df['Amount'].describe())

            except Exception as e:
                logging.error(f"Error during univariate analysis for 'Amount': {e}")



        # Check if 'Class' exists in the dataset for a bar plot
        if 'Class' in self.df.columns:
            try:
                logging.info("Performing bar plot for 'Class' column")
                print("\nUnivariate Analysis for: Class")

                # Bar plot for 'Class' column
                plt.figure(figsize=(10, 6))
                sns.countplot(x=self.df['Class'], order=self.df['Class'].value_counts().index)  # Order by counts
                plt.title("Class Distribution")
                plt.xlabel('Class')
                plt.ylabel('Count')
                plt.show()  # Ensure the plot is displayed

                # Display value counts for 'Class'
                print(self.df['Class'].value_counts())

            except Exception as e:
                logging.error(f"Error during univariate analysis for 'Class': {e}")


        # Perform univariate analysis on other suitable columns
        suitable_columns = self.get_suitable_columns()

        # Remove 'Amount' and 'Class' from the list since they have already been processed
        suitable_columns = [col for col in suitable_columns if col not in ['Amount', 'Class']]

        # Process other columns for univariate analysis
        for col in suitable_columns:
            try:
                logging.info(f"Performing univariate analysis for column: {col}")
                print(f"\nUnivariate Analysis for: {col}")

                if self.df[col].dtype in ['int64', 'float64']:
                    # Numeric columns - plot distribution
                    plt.figure(figsize=(10, 6))
                    sns.histplot(self.df[col], kde=True)
                    plt.title(f"Distribution of {col}")
                    plt.xlabel(col)
                    plt.ylabel('Frequency')
                    plt.show()

                    # Display summary statistics
                    print(self.df[col].describe())
                else:
                    # Categorical columns - plot value counts
                    plt.figure(figsize=(10, 6))
                    sns.countplot(x=self.df[col], order=self.df[col].value_counts().index)  # Order by counts
                    plt.title(f"Count plot of {col}")
                    plt.xlabel(col)
                    plt.ylabel('Count')
                    plt.show()

                    # Display value counts
                    print(self.df[col].value_counts())

            except Exception as e:
                logging.error(f"Error during univariate analysis for column {col}: {e}")

    def bivariate_analysis(self):
        """
        Perform bivariate analysis between key features in the dataset.
        Includes analysis on 'Amount', 'Class', and other relevant features.
        """
        logging.info("Starting bivariate analysis")
        
        dataset_name = os.path.basename(self.filepath).lower()
        
        try:
            # Bivariate analysis for Fraud_Data.csv
            if 'fraud' in dataset_name:
                if 'purchase_value' in self.df.columns and 'class' in self.df.columns:
                    plt.figure(figsize=(10, 6))
                    sns.boxplot(x='class', y='purchase_value', data=self.df)
                    plt.title('Boxplot of Purchase Value by Class')
                    plt.ylabel('Purchase Value')
                    plt.xlabel('Class')
                    plt.show()

                # Analyze age distribution by fraud class
                if 'age' in self.df.columns and 'class' in self.df.columns:
                    plt.figure(figsize=(10, 6))
                    sns.boxplot(x='class', y='age', data=self.df)
                    plt.title('Boxplot of Age by Fraud Class')
                    plt.ylabel('Age')
                    plt.xlabel('Class')
                    plt.show()

                # Analyzing time-based patterns if the dataset contains timestamps
                if 'purchase_time' in self.df.columns and 'class' in self.df.columns:
                    self.df['hour_of_day'] = self.df['purchase_time'].dt.hour
                    plt.figure(figsize=(10, 6))
                    sns.countplot(x='hour_of_day', hue='class', data=self.df)
                    plt.title('Transactions by Hour of Day and Fraud Class')
                    plt.ylabel('Count')
                    plt.xlabel('Hour of Day')
                    plt.legend(title='Class')
                    plt.show()

            # Bivariate analysis for creditcard.csv
            elif 'credit' in dataset_name:
                if 'Amount' in self.df.columns and 'Class' in self.df.columns:
                    plt.figure(figsize=(10, 6))
                    sns.boxplot(x='Class', y='Amount', data=self.df)
                    plt.title('Boxplot of Amount by Class')
                    plt.ylabel('Amount')
                    plt.xlabel('Class')
                    plt.show()
                    
                    # Calculate correlation between Amount and Class
                    correlation = self.df[['Amount', 'Class']].corr().iloc[0, 1]
                    print(f"Correlation between Amount and Class: {correlation:.2f}")

                    correlation_matrix = self.df[['Amount', 'Class']].corr()

                    plt.figure(figsize=(8, 6))
                    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
                    plt.title('Correlation Heatmap Between Amount and Class')
                    plt.show()

                    # Visualize the relationship between Amount and Class
                    plt.figure(figsize=(10, 6))
                    sns.scatterplot(x='Amount', y='Class', data=self.df)
                    plt.title('Scatter Plot of Amount vs Class')
                    plt.xlabel('Amount')
                    plt.ylabel('Class')
                    plt.show()
        except Exception as e:
            logging.error(f"Error during bivariate analysis: {e}")

                
