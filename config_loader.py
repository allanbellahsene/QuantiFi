import os
import yaml
from datetime import datetime

def load_config(config_file='config.yaml'):
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load the YAML configuration
    with open(os.path.join(current_dir, config_file), 'r') as file:
        config = yaml.safe_load(file)

    # Convert date strings to datetime objects
    config['backtest']['start_date'] = datetime.strptime(config['backtest']['start_date'], '%Y-%m-%d')
    config['backtest']['end_date'] = datetime.strptime(config['backtest']['end_date'], '%Y-%m-%d')
    
    # Add the directory configurations
    config['BASE_DIR'] = current_dir
    config['DATA_DIR'] = os.path.join(current_dir, '..', 'DATA')
    config['MARKET_CAP_DIR'] = os.path.join(config['DATA_DIR'], 'Crypto_Market_Cap')
    
    return config

# Load the configuration
CONFIG = load_config()

# You can access the configuration using CONFIG dictionary
# For example:
# print(CONFIG['DATA_DIR'])
# print(CONFIG['params']['transaction_cost'])