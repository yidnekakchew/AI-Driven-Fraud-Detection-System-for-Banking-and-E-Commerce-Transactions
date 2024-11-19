# Data Preprocessing and Analysis Tool

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![Pandas](https://img.shields.io/badge/Pandas-1.0%2B-brightgreen)
![NumPy](https://img.shields.io/badge/NumPy-1.18%2B-yellow)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.2%2B-orange)
![Seaborn](https://img.shields.io/badge/Seaborn-0.10%2B-red)

## Overview

This script provides a comprehensive tool for data preprocessing and analysis, particularly useful for datasets related to fraud detection and credit card transactions. It offers functionality for data loading, cleaning, univariate analysis, and bivariate analysis, with built-in logging for easy tracking of operations.

## Features

- **Data Loading**: Automatically loads CSV files into pandas DataFrames.
- **Data Overview**: Provides dataset structure, preview, data types, and missing value information.
- **Data Cleaning**: Removes duplicates and corrects data types for specific columns.
- **Univariate Analysis**: Performs analysis on individual columns, generating appropriate visualizations.
- **Bivariate Analysis**: Analyzes relationships between key features in the dataset.
- **Logging**: Maintains detailed logs of all operations for easy tracking and debugging.

## Requirements

- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- IPython

## folder structure
```
├── scripts/
│   ├──__init__.py
│   ├── data_preprocessing.py
│   └── feature_engineering.py
│   ├──merge_dataset.py
│   ├──README.md
```
## Usage

1. Import the `pre_processing` class from the script:

   ```python
   from data_preprocessing_script import pre_processing

- Create an instance of the class with your CSV file path:

- preprocessor = pre_processing('path/to/your/dataset.csv')
Use the various methods to analyze your data:

```
preprocessor.data_overview()
preprocessor.data_cleaning()
preprocessor.univariate_analysis()
preprocessor.bivariate_analysis()
```

## Key Methods
- `data_overview()`: Displays dataset shape, preview, data types, and missing values.
- `data_cleaning()`: Removes duplicates and corrects data types.
- `univariate_analysis()`: Performs analysis on individual columns.
- `bivariate_analysis()`: Analyzes relationships between key features.

## Logging
The script automatically creates log files in a logging directory within the parent directory of the current working directory. Log files are named based on the dataset being processed.

## Notes
- The script is optimized for datasets related to fraud detection and credit card transactions but can be adapted for other types of data.
- A total of three datasets were used for this project: fraud detection, credit card transactions, and IP address information.
- Visualizations are automatically generated and displayed during analysis.



