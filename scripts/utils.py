import pandas as pd

def load_csv_file(file_path):
    """
    Load a CSV file into a pandas DataFrame.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: The loaded DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None
    
def clean_data(df):
    """
    Clean the DataFrame by handling missing values and duplicates.

    Parameters:
    df (pd.DataFrame): The DataFrame to clean.

    Returns:
    pd.DataFrame: The cleaned DataFrame.
    """
    df = df.drop_duplicates()
    df = df.dropna()
    return df

def explore_data(df):
    """
    Explore the DataFrame by providing basic statistics and information.

    Parameters:
    df (pd.DataFrame): The DataFrame to explore.

    Returns:
    None
    """
    print("DataFrame Head:")
    print(df.head())
    print("\nDataFrame Info:")
    print(df.info())
    print("\nDataFrame Description:")
    print(df.describe())