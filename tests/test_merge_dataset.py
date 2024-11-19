import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from scripts.merge_dataset import MergeDataset

class TestMergeDataset(unittest.TestCase):

    @patch('pandas.read_csv')
    def test_initialization_success(self, mock_read_csv):
        # Mock the return value of read_csv
        mock_read_csv.side_effect = [
            pd.DataFrame({'ip_address': [123456789]}),  # Mock fraud data
            pd.DataFrame({'lower_bound_ip_address': [123456780], 'upper_bound_ip_address': [123456790], 'country': ['CountryA']})  # Mock IP data
        ]
        
        merge_dataset = MergeDataset('dummy_path1.csv', 'dummy_path2.csv')
        self.assertIsNotNone(merge_dataset.fraud_data)
        self.assertIsNotNone(merge_dataset.ip_data)

    @patch('pandas.read_csv')
    def test_file_not_found(self, mock_read_csv):
        mock_read_csv.side_effect = FileNotFoundError

        with self.assertRaises(FileNotFoundError):
            MergeDataset('invalid_path1.csv', 'invalid_path2.csv')

    @patch('pandas.read_csv')
    def test_empty_data_error(self, mock_read_csv):
        mock_read_csv.side_effect = pd.errors.EmptyDataError

        with self.assertRaises(pd.errors.EmptyDataError):
            MergeDataset('empty_file1.csv', 'empty_file2.csv')

    @patch('pandas.read_csv')
    def test_match_ip_to_country(self, mock_read_csv):
        mock_read_csv.side_effect = [
            pd.DataFrame({'ip_address': [123456789]}),
            pd.DataFrame({
                'lower_bound_ip_address': [123456780],
                'upper_bound_ip_address': [123456790],
                'country': ['CountryA']
            })
        ]
        
        merge_dataset = MergeDataset('dummy_path1.csv', 'dummy_path2.csv')
        merge_dataset.match_ip_to_country()
        
        # Check if the country is correctly assigned
        self.assertEqual(merge_dataset.fraud_data['country'].iloc[0], 'CountryA')

    @patch('pandas.read_csv')
    @patch('pandas.DataFrame.to_csv')
    def test_save_merged_data(self, mock_to_csv, mock_read_csv):
        mock_read_csv.side_effect = [
            pd.DataFrame({'ip_address': [123456789]}),
            pd.DataFrame({
                'lower_bound_ip_address': [123456780],
                'upper_bound_ip_address': [123456790],
                'country': ['CountryA']
            })
        ]
        
        merge_dataset = MergeDataset('dummy_path1.csv', 'dummy_path2.csv')
        merge_dataset.save_merged_data('dummy_save_path.csv')

        # Check if to_csv was called
        mock_to_csv.assert_called_once_with('dummy_save_path.csv', index=False)

if __name__ == '__main__':
    unittest.main()