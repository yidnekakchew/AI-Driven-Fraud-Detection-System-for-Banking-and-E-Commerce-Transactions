import os
import logging
import pandas as pd

# Get the current working directory
current_directory = os.getcwd()

# Ensure the logging directory exists
log_directory = os.path.join(current_directory, 'logging')
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

log_file = os.path.join(log_directory, "merge_dataset.log")
logging.basicConfig(filename=log_file,
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filemode='a')


class MergeDataset:
    def __init__(self, file_path1, file_path2):
        try:
            logging.info(f"Loading datasets from {file_path1} and {file_path2}")
            self.fraud_data = pd.read_csv(file_path1)
            self.ip_data = pd.read_csv(file_path2)
            logging.info("Successfully loaded the datasets")
        except FileNotFoundError as e:
            logging.error(f"File not found: {e}")
            raise
        except pd.errors.EmptyDataError as e:
            logging.error(f"Empty data error: {e}")
            raise
        except Exception as e:
            logging.error(f"Error loading datasets: {e}")
            raise

    def match_ip_to_country(self):
        logging.info("Matching IP addresses to countries.")
        try:
            # Convert IP addresses in fraud_data to integer
            self.fraud_data['ip_address'] = self.fraud_data['ip_address'].astype(float).astype(int)

            # Function to find the country based on IP address range
            def find_country(ip):
                country_row = self.ip_data[(self.ip_data['lower_bound_ip_address'] <= ip) & (self.ip_data['upper_bound_ip_address'] >= ip)]
                if not country_row.empty:
                    return country_row['country'].values[0]
                return 'Unknown'

            # Apply the matching function to fraud_data
            self.fraud_data['country'] = self.fraud_data['ip_address'].apply(find_country)

            logging.info("IP address matching completed successfully.")
        except Exception as e:
            logging.error(f"Error during IP address matching: {e}")
            raise

    def save_merged_data(self, save_location):
        # Save the merged DataFrame to a CSV file and log the operation
        try:
            self.match_ip_to_country()
            self.fraud_data.to_csv(save_location, index=False)
            logging.info(f"Merged data saved to {save_location}.")
            print(f"Merged data saved to {save_location}")
        except Exception as e:
            logging.error(f"Error saving merged data: {e}")
            raise


if __name__ == "__main__":
    file_path1 = "data/Processed_Fraud_Data.csv"
    file_path2 = "data/Processed_IpAddress_data.csv"
    save_location = "data/Merged_FIp_data.csv"

    try:
        merge_dataset = MergeDataset(file_path1, file_path2)
        merge_dataset.save_merged_data(save_location)
    except Exception as e:
        print(f"An error occurred: {e}")
