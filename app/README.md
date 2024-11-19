# Fraud Detection Dashboard

## Overview

The **Fraud Detection Dashboard** is a web application that visualizes and monitors fraud detection data in real time. It consists of a REST API built with Flask to serve fraud-related data and a dashboard interface built with Dash for interactive visualization. This application provides insights into transaction patterns, fraud distribution across devices and browsers, and fraud trends over time.

## Folder Structure
```
├── app/ 
│   ├── app.py            # Main Flask API application
│   ├── dashboard.py      # Dash application for data visualization
│   ├── __init__.py       # Initialization file for the app package
│   └── requirements.txt   # Dependencies for the application
├── README.md             # Project documentation
└── .gitignore            # Git ignore configuration
```

## Features

1. **Summary Statistics**  
   Provides an overview of total transactions, total fraud cases, and the percentage of fraudulent transactions.

2. **Fraud Cases Over Time**  
   Displays a time series chart showing the number of fraud cases over different months.

3. **Top Devices by Fraud Cases**  
   Shows the devices most frequently associated with fraudulent activities.

4. **Fraud Cases by Browser**  
   Visualizes fraud occurrences by browser.

5. **Real-Time Updates**  
   Data refreshes every 30 seconds, ensuring up-to-date monitoring.

## Getting Started

### Installation

1. **Clone the repository** and checkout the 5th branch.

   ```bash
   git clone https://github.com/Endekalu777/Enhanced-Fraud-Detection-for-E-commerce-and-Banking-Transactions.git
   cd Enhanced-Fraud-Detection-for-E-commerce-and-Banking-Transactions
   git checkout task5
   ```
### install dependencies

```
   pip install -r requirements.txt
```

### Place your data
Ensure the fraud data file Fraud_Data.csv is located in the data folder.

## Running the application
- Start the Flask API

```
python app/app.py
```
- The API will run on http://localhost:5000.

## Start the Dash Dashboard
- In a new terminal, run:
```
python app/dashboard.py
```
- The dashboard will run on http://localhost:8050.

# API Endpoints

- **GET /api/summary**
  - Returns a summary of total transactions, total fraud cases, and the fraud percentage.

- **GET /api/time-series**
  - Provides fraud cases grouped by month and year.

- **GET /api/device-stats**
  - Retrieves information on the top 10 devices associated with fraud cases.

- **GET /api/browser-stats**
  - Presents fraud cases grouped by browser.

## License

This project is licensed under the MIT License. Refer to the LICENSE file for details.

## Author

Endekalu Simon

## Acknowledgments

This project utilizes Dash for the dashboard interface and Flask for the backend API. Special thanks to 10academy for providing the dataset and challenge for API design.

*Note: This represents the 5th branch of the project, which includes specific features for the Fraud Detection Dashboard.* 😊