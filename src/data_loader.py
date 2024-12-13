import pandas as pd


# load data
def load_data(file_path):
    """Load a CSV file and return a DataFrame."""
    return pd.read_csv(file_path)

# load all company datas
def load_all_data():
    """Load datasets for all companies."""
    files = {
        'AAPL': '..yfinance_data/AAPL_historical_data.csv',
        'AMZN':'../yfinance_data/AMZN_historical_data.csv',
        'GOOG': '../yfinance_data/GOOG_historical_data.csv',
        'META':'../yfinance_data/META_historical_data.csv',
        'MSFT': '../yfinance_data/MSFT_historical_data.csv',
        'NVDA': '../yfinance_data/NVDA_historical_data.csv',
        'TSLA': '../yfinance_data/TSLA_historical_data.csv'
    }
    data = {name: load_data(path) for name, path in files.items()}
    return data