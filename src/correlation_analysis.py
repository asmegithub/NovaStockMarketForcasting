import pandas as pd
import os
import matplotlib.pyplot as plt
from textblob import TextBlob
import seaborn as sns

class CorrelationAnalyzer:
    def __init__(self, file_path=None, folder_path=None):
        """
        Initializes the StockPriceAnalyzer class with a CSV file path or a folder path.
        
        param file_path: str (optional) - Path to a single CSV file containing stock data.
        param folder_path: str (optional) - Path to a folder containing multiple CSV files to be merged.
        """
        self.file_path = file_path
        self.folder_path = folder_path
        self.data = pd.DataFrame()
        
        if self.file_path:
            # Load data from a single CSV file
            self.data = pd.read_csv(file_path)
        elif self.folder_path:
            # Automatically merge CSV files from the specified folder
            self.merge_csv_files(folder_path)
    def load_data(self, file_path):
        self.file_path = file_path
        self.data = pd.read_csv(file_path)
        return self.data    
    def merge_csv_files(self, folder_path):
        """
        Merges all CSV files from the specified folder into one DataFrame and adds a 'stock_symbol' column.
        
        folder_path: str - Path to the folder containing CSV files.
        return: DataFrame - Merged DataFrame of all CSV files in the folder.
        """
        merged_df = pd.DataFrame()
        for file in os.listdir(folder_path):
            if file.endswith('.csv'):
                file_path = os.path.join(folder_path, file)
                df = pd.read_csv(file_path)
                
                # Extract stock symbol from the file name before the underscore
                stock_symbol = os.path.splitext(file)[0].split('_')[0]  # Extract symbol before the underscore
                
                # Add the stock_symbol column to the DataFrame
                df['stock_symbol'] = stock_symbol
                
                # Merge the DataFrame into the main DataFrame
                merged_df = pd.concat([merged_df, df], ignore_index=True)
        
        self.data = merged_df
        return self.data
    def calculate_daily_returns(self,price):
        daily_returns = price.pct_change()
        return daily_returns
    
    
    def correlation_matrix(self,data=None):
         # Use provided data or fall back to the internal DataFrame
        if data is None:
            data = self.data
        # Compute the correlation matrix
        self.data.drop(columns=['Stock Splits','Dividends'], axis=1, inplace=True)        
        corr_matrix = self.data.select_dtypes(include=['int', 'float']).corr()

        # Plot correlation using heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, cmap='coolwarm', annot=True,fmt=".2f", cbar=True, linewidths=0.5)
        plt.show()
    def correlation_each_symbol(self, data=None):
        # Use provided data or fall back to the internal DataFrame
        if data is None:
            data = self.data

        # Get unique stock symbols
        unique_symbols = data['Ticker_symbol'].unique()
        # Initialize a list to store correlation results
        correlation_results = []

        # Loop through each unique stock symbol
        for symbol in unique_symbols:
            # Filter data for the current symbol
            symbol_data = data[data['Ticker_symbol'] == symbol]
            
            # Calculate the correlation between 'polarity' and 'daily_return'
            correlation = symbol_data[['polarity', 'daily_return']].corr().loc['polarity', 'daily_return']
            
            # Append the result to the list
            correlation_results.append({'Ticker_symbol': symbol, 'correlation': correlation})

        # Convert the results list to a DataFrame
        correlation_df = pd.DataFrame(correlation_results)

        # Return the correlation DataFrame
        return correlation_df
    def visualize_relationships(self,aggregated_data=None):
        """
        Visualizes scatter plots to show relationships between Polarity and other financial metrics:
        Daily Return, Close Price, and Volume.
        
        Parameters:
            aggregated_data (DataFrame): Pandas DataFrame containing 'polarity', 'daily_return', 'Close', and 'Volume'.
        """
        # Set up the figure size
        plt.figure(figsize=(14, 10))
        
        # Scatter plot for Polarity vs. Daily Return
        plt.subplot(2, 2, 1)
        sns.scatterplot(x='polarity', y='daily_return', data=aggregated_data)
        plt.title('Polarity vs. Daily Return')
        plt.xlabel('Polarity')
        plt.ylabel('Daily Return')

        # Scatter plot for Polarity vs. Close Price
        plt.subplot(2, 2, 2)
        sns.scatterplot(x='polarity', y='Close', data=aggregated_data)
        plt.title('Polarity vs. Close Price')
        plt.xlabel('Polarity')
        plt.ylabel('Close Price')

        # Scatter plot for Polarity vs. Volume
        plt.subplot(2, 2, 3)
        sns.scatterplot(x='polarity', y='Volume', data=aggregated_data)
        plt.title('Polarity vs. Volume')
        plt.xlabel('Polarity')
        plt.ylabel('Volume')

        # Adjust layout and display the plots
        plt.tight_layout()
        plt.show()
    def visualize_time_series(self,merged_data):
        """
        Visualizes Polarity and Stock Prices over time, along with their rolling averages.
        
        Parameters:
            merged_data (DataFrame): Pandas DataFrame containing 'Date', 'polarity', 'Close', and 'Open'.
            rolling_window (int): Window size for calculating rolling averages (default: 30 days).
        """
        # Ensure 'Date' is in datetime format
        merged_data['Date'] = pd.to_datetime(merged_data['Date'])
        
        # Sort data by Date
        merged_data = merged_data.sort_values(by='Date')
        
        # Set Date as index for easier plotting
        merged_data.set_index('Date', inplace=True)
        
        # Plot Polarity and Stock Prices
        plt.figure(figsize=(14, 7))
        
        # Plot Polarity
        plt.subplot(2, 1, 1)
        plt.plot(merged_data.index, merged_data['polarity'], label='Polarity', color='blue')
        plt.title('Polarity Over Time')
        plt.xlabel('Date')
        plt.ylabel('Polarity')
        plt.legend()
        
        # Plot Stock Prices
        plt.subplot(2, 1, 2)
        plt.plot(merged_data.index, merged_data['Close'], label='Close Price', color='green', alpha=0.7)
        plt.plot(merged_data.index, merged_data['Open'], label='Open Price', color='red', alpha=0.7)
        plt.title('Stock Prices Over Time')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Calculate Rolling Averages
        merged_data['Close_rolling_mean'] = merged_data['Close'].rolling(window=30).mean()
        merged_data['Polarity_rolling_mean'] = merged_data['polarity'].rolling(window=30).mean()
        
        # Plot Rolling Averages
        plt.figure(figsize=(14, 7))
        plt.plot(merged_data.index, merged_data['Close'], label='Close Price', color='green', alpha=0.5)
        plt.plot(merged_data.index, merged_data['Close_rolling_mean'], label='Rolling Mean Close Price', color='darkgreen')
        plt.plot(merged_data.index, merged_data['Polarity_rolling_mean'], label='Rolling Mean Polarity', color='blue')
        plt.title('Rolling Averages of Polarity and Stock Prices')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.show()
    
    def Plot_correlation_with_symbol(self, data=None):
        # Use provided data or fall back to the internal DataFrame
        if data is None:
            data = self.data

        # Get unique stock symbols
        unique_symbols = data['Ticker_symbol'].unique()

        # Scatter Plot for Polarity vs. Daily Return
        plt.figure(figsize=(12, 6))

        # Loop through each stock symbol and plot
        for symbol in unique_symbols:
            symbol_data = data[data['Ticker_symbol'] == symbol]
            plt.scatter(symbol_data['polarity'], symbol_data['daily_return'], label=symbol, alpha=0.5)

        plt.title('Scatter Plot of Polarity vs. Daily Return')
        plt.xlabel('Polarity')
        plt.ylabel('Daily Return')
        plt.legend(title='Ticker Symbol', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Generate correlation DataFrame
        correlation_df = self.correlation_each_symbol(data)

        # Bar Plot of Correlations
        plt.figure(figsize=(12, 6))

        # Plot bar chart
        sns.barplot(x='Ticker_symbol', y='correlation', data=correlation_df, palette='viridis')

        plt.title('Correlation between Polarity and Daily Return by Stock Symbol')
        plt.xlabel('Ticker Symbol')
        plt.ylabel('Correlation')
        plt.xticks(rotation=90)  # Rotate x labels for better readability
        plt.grid(True)
        plt.tight_layout()
        plt.show()