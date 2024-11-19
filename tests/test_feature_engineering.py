import unittest
from unittest.mock import patch
import pandas as pd
from scripts.feature_engineering import FeatureEngineering

class TestFeatureEngineering(unittest.TestCase):

    @patch('pandas.read_csv')
    def setUp(self, mock_read_csv):
        """Set up mock data for testing."""
        # Mock datasets
        mock_fraud_data = pd.DataFrame({
            'user_id': [1, 1, 2, 3, 3, 4],
            'purchase_value': [34.0, 50.0, 16.0, 44.0, 20.0, 100.0],
            'purchase_time': [
                '2023-01-01 10:00:00',
                '2023-01-01 11:00:00',
                '2023-01-02 10:00:00',
                '2023-01-02 11:00:00',
                '2023-01-03 12:00:00',
                '2023-01-04 14:00:00'
            ],
            'browser': ['Chrome', 'Firefox', 'Chrome', 'Safari', 'Chrome', 'Firefox'],
            'sex': ['Male', 'Female', 'Female', 'Male', 'Male', 'Female'],
            'country': ['USA', 'Canada', 'USA', 'USA', 'Canada', 'USA']
        })

        mock_creditcard_data = pd.DataFrame({
            'user_id': [1, 2, 3, 4],
            'Amount': [5000, 3000, 4000, 7000],
            'balance': [2500, 1500, 2000, 500]
        })

        # Set the return value of read_csv
        mock_read_csv.side_effect = [mock_fraud_data, mock_creditcard_data]

        # Initialize the FeatureEngineering object
        self.feature_engineering = FeatureEngineering('dummy_fraud_file.csv', 'dummy_creditcard_file.csv')

    def test_compute_transaction_features(self):
        self.feature_engineering.compute_transaction_features()
        fraud_df = self.feature_engineering.fraud_df
        
        # Check if new columns are created
        self.assertIn('transaction_frequency', fraud_df.columns)
        self.assertIn('time_diff', fraud_df.columns)
        self.assertIn('hour_of_day', fraud_df.columns)
        self.assertIn('day_of_week', fraud_df.columns)

        # Validate transaction frequency for a specific user_id
        self.assertEqual(fraud_df.loc[fraud_df['user_id'] == 1, 'transaction_frequency'].iloc[0], 2)

    def test_scale_features(self):
        self.feature_engineering.compute_transaction_features()
        self.feature_engineering.scale_features()
        
        fraud_df = self.feature_engineering.fraud_df
        
        # Check if features are scaled and contain no NaN values
        self.assertFalse(fraud_df[['purchase_value', 'transaction_frequency', 'time_diff', 'hour_of_day']].isnull().values.any())

    def test_encode_categorical_features(self):
        self.feature_engineering.compute_transaction_features()
        self.feature_engineering.encode_categorical_features()
        
        fraud_df = self.feature_engineering.fraud_df
        
        # Check if one-hot encoding has been applied
        self.assertIn('browser_Firefox', fraud_df.columns)
        self.assertIn('browser_Safari', fraud_df.columns)
        self.assertIn('sex_Male', fraud_df.columns)
        self.assertIn('country_USA', fraud_df.columns)

    def test_save_cleaned_data(self):
        output_fraud_file = 'cleaned_fraud.csv'
        output_creditcard_file = 'cleaned_creditcard.csv'
        
        self.feature_engineering.save_cleaned_data(output_fraud_file, output_creditcard_file)
        
        # Validate if files are created and contain data
        self.assertTrue(pd.read_csv(output_fraud_file).shape[0] > 0)
        self.assertTrue(pd.read_csv(output_creditcard_file).shape[0] > 0)

    @classmethod
    def tearDownClass(cls):
        # Optional: Cleanup output files after tests
        import os
        if os.path.exists('cleaned_fraud.csv'):
            os.remove('cleaned_fraud.csv')
        if os.path.exists('cleaned_creditcard.csv'):
            os.remove('cleaned_creditcard.csv')

if __name__ == '__main__':
    unittest.main()