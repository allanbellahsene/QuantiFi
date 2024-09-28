import os
from datetime import datetime, timedelta
from config import MARKET_CAP_DIR, PARAMS as params
import pandas as pd
import pickle

class Utils:
    @staticmethod
    def convert_dollar_to_float(value):
        value = value.replace('$', '').replace(',', '')
        return float(value)

    @staticmethod
    def convert_pct_to_float(value):
        value = value.replace('--', '0').replace('<', '').replace('>', '').replace('%', '').replace(',', '')
        return float(value)/100

    @staticmethod
    def import_market_cap_files(n_weeks=1):
        """
        Import market cap files from a specified directory.
        :param n_weeks: Number of weeks between each file to keep (default is 1)
        :return: List of file names to keep
        """
        # Get all files in the specified directory
        files = [f for f in os.listdir(MARKET_CAP_DIR) if f.startswith('market_cap_snapshot_') and f.endswith('.csv')]

        # Sort the files by date
        files.sort()

        if not files:
            print(f"No market cap files found in {MARKET_CAP_DIR}")
            return []

        # Parse the date of the first file
        first_date = datetime.strptime(files[0].split('_')[-1].split('.')[0], '%Y%m%d')

        # Initialize the list of files to keep
        files_to_keep = [files[0]]  # Always keep the first file

        # Set the interval
        interval = timedelta(weeks=n_weeks)

        # Iterate through the rest of the files
        for file in files[1:]:
            file_date = datetime.strptime(file.split('_')[-1].split('.')[0], '%Y%m%d')
            if file_date >= first_date + len(files_to_keep) * interval:
                files_to_keep.append(file)

        return [os.path.join(MARKET_CAP_DIR, file) for file in files_to_keep]

    @staticmethod
    def check_file_exists_os(directory, filename):
        """
        Check if a file exists in the specified directory using os.path.
        
        :param directory: The directory to check
        :param filename: The name of the file to check for
        :return: True if the file exists, False otherwise
        """
        file_path = os.path.join(directory, filename)
        return os.path.isfile(file_path)

    @staticmethod
    def save_list(list_object, filename):
        """
        Save a list object using pickle.
        
        :param list_object: The list to save
        :param filename: The name of the file to save to
        """
        with open(filename, 'wb') as f:
            pickle.dump(list_object, f)

    @staticmethod
    def load_list(filename):
        """
        Load a list object using pickle.
        
        :param filename: The name of the file to load from
        :return: The loaded list object
        """
        with open(filename, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def save_dataframe_pickle(df, filename):
        """
        Save a pandas DataFrame using pickle.
        
        :param df: pandas DataFrame to save
        :param filename: Name of the file to save to
        """
        with open(filename, 'wb') as f:
            pickle.dump(df, f)

    @staticmethod
    def load_dataframe_pickle(filename):
        """
        Load a pandas DataFrame using pickle.
        
        :param filename: Name of the file to load from
        :return: pandas DataFrame
        """
        with open(filename, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def extract_date(filename):
        # Split the filename by underscore and take the last part
        date_part = filename.split('_')[-1]
        # Remove the .csv extension
        date_part = date_part.split('.')[0]
        return pd.to_datetime(date_part, format='%Y%m%d').date()

    @staticmethod
    def flatten_and_unique(list_of_lists):
        # Step 1: Flatten the list of lists
        flattened = [item for sublist in list_of_lists for item in sublist]

        # Step 2: Keep only unique elements
        unique_elements = list(dict.fromkeys(flattened))

        return unique_elements


if __name__ == '__main__':

    utils = Utils()
    files_to_keep = utils.import_market_cap_files()
    N = len(files_to_keep)
    min_volume = params['min_volume_thres']
    max_MC = params['max_MC_rank']

    SYMBOLS = []

    for file in files_to_keep:

        mc = pd.read_csv(file)
        date = utils.extract_date(file)

        mc['volume'] = mc['volume (24h)'].apply(utils.convert_dollar_to_float)
        mc['7d_perf'] = mc['% 7d'].apply(utils.convert_pct_to_float)
        symbols = mc.loc[(mc.Rank<=max_MC) & (mc.volume >= min_volume)].Symbol.unique()

        symbols = [s for s in symbols if 'US' not in s] #remove stable coins
        symbols = [s for s in symbols if 'DAI' not in s]
        ASSET_UNIVERSE = [s + '-USD' for s in symbols]

        SYMBOLS.append(ASSET_UNIVERSE)

    SYMBOLS = utils.flatten_and_unique(SYMBOLS)
    print(SYMBOLS)