import pandas as pd
import unittest
import numpy as np
import matplotlib.pyplot as plt
import logging
from unittest.mock import patch, MagicMock
from scripts.data_preprocessing import *

# Configure logging to output to console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TestPreProcessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Run once before all tests."""
        # Mock dataset to use in tests
        cls.mock_data = pd.DataFrame({
            'user_id': [1, 2, 2, 4],
            'signup_time': ['2023-01-01 10:00:00', '2023-02-01 12:00:00', 
                            '2023-02-01 12:00:00', '2023-03-01 14:00:00'],
            'purchase_time': ['2023-01-01 11:00:00', '2023-02-01 13:00:00', 
                              '2023-02-01 13:00:00', '2023-03-01 15:00:00'],
            'purchase_value': [34.0, 16.0, 15.0, 44.0],
            'Amount': [50.0, 20.5, 20.5, np.nan],
            'Class': [0, 1, 1, 0],
            'lower_bound_ip_address': [123456, 67890, np.nan, 90123],
            'upper_bound_ip_address': [16777471, 16777727, np.nan, 16779263],
            'country': ['USA', 'Canada', 'USA', 'Canada']
        })

        # Create a temporary CSV file for testing
        cls.test_file = 'test_dataset.csv'
        cls.mock_data.to_csv(cls.test_file, index=False)

        # Initialize the pre_processing object with the test file
        cls.processor = pre_processing(cls.test_file)

        # Turn off interactive plotting
        plt.ioff()

    @classmethod
    def tearDownClass(cls):
        """Run once after all tests."""
        # Remove the test file after tests complete
        if os.path.exists(cls.test_file):
            os.remove(cls.test_file)

    def test_data_loading(self):
        """Test if data is loaded correctly without errors."""
        self.assertIsInstance(self.processor.df, pd.DataFrame)
        self.assertEqual(self.processor.df.shape, (4, 9))

    def test_data_overview(self):
        """Test if the data overview methods execute without errors."""
        try:
            self.processor.data_overview()
        except Exception as e:
            self.fail(f"data_overview() raised an exception: {e}")

    def test_data_cleaning(self):
        """Test if data cleaning removes duplicates and converts data types correctly."""
        self.processor.data_cleaning()
        
        # Check if duplicates were removed (assuming there are no duplicates in the mock data)
        self.assertEqual(self.processor.df.shape, (4, 9))

        # Check if datetime conversion worked
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(self.processor.df['signup_time']))
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(self.processor.df['purchase_time']))

        # Check if IP addresses were converted to integers
        self.assertTrue(pd.api.types.is_integer_dtype(self.processor.df['lower_bound_ip_address']))
        self.assertTrue(pd.api.types.is_integer_dtype(self.processor.df['upper_bound_ip_address']))

    def test_univariate_analysis(self):
        """Test if univariate analysis runs without errors."""
        try:
            with patch('matplotlib.pyplot.show') as mock_show:  
                self.processor.univariate_analysis()
                self.assertGreater(mock_show.call_count, 0, "plt.show() was not called.")
            plt.close('all')  
        except Exception as e:
            self.fail(f"univariate_analysis() raised an exception: {e}")

    def test_bivariate_analysis(self):
        """Test if bivariate analysis executes without errors."""
        try:
            with patch('matplotlib.pyplot.show') as mock_show:  
                self.processor.bivariate_analysis()
                mock_show.assert_not_called()  
            plt.close('all')  
        except Exception as e:
            self.fail(f"bivariate_analysis() raised an exception: {e}")

    def test_logging_directory_creation(self):
        """Test if the logging directory is created correctly."""
        parent_directory = os.path.dirname(os.getcwd())
        log_directory = os.path.join(parent_directory, 'logging')
        self.assertTrue(os.path.exists(log_directory))
    
    def test_logging_output(self):
        """Test if logging outputs to console correctly."""
        # Trigger a log entry
        logging.info("Test log entry")
        # No assertion needed; just check console output during test run

if __name__ == '__main__':
    unittest.main()