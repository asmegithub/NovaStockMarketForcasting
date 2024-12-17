# Nova Stock Market Forecasting

## Overview
This project aims to forecast stock market trends using machine learning algorithms. We will analyze historical stock data and build predictive models to assist in making informed investment decisions.

## Objectives
- Collect and preprocess historical stock market data
- Explore and visualize the data to identify patterns and trends
- Develop and evaluate machine learning models for stock price prediction
- Implement the best-performing model for real-time forecasting

## Requirements
All dependencies are listed in the `requirements.txt` file. The main requirements include:
- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## Installation
1. Clone the repository:
    ```bash
    git clone /home/asmarez/projects/AI/week1/NovaStockMarketForcasting
    ```
2. Navigate to the project directory:
    ```bash
    cd NovaStockMarketForcastin
    g
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Progress
- Data collection and preprocessing: In progress
- Data exploration and visualization: Pending
- Model development and evaluation: Pending
- Real-time forecasting implementation: Pending

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

## License
This project is licensed under 10 Acadamy 

## Business Objective
Nova Financial Solutions aims to enhance its predictive analytics capabilities to significantly boost its financial forecasting accuracy and operational efficiency through advanced data analysis. This project focuses on:

### Sentiment Analysis
Perform sentiment analysis on the headline text to quantify the tone and sentiment expressed in financial news. This involves using natural language processing (NLP) techniques to derive sentiment scores, which can be associated with the respective Stock Symbol to understand the emotional context surrounding stock-related news.

### Correlation Analysis
Establish statistical correlations between the sentiment derived from news articles and the corresponding stock price movements. This involves tracking stock price changes around the date the article was published and analyzing the impact of news sentiment on stock performance. Consideration is given to publication dates and potentially times if such data is available.

### Recommendations
Leverage insights from this sentiment analysis to suggest investment strategies. These strategies utilize the relationship between news sentiment and stock price fluctuations to predict future movements. The final report provides clear, actionable insights based on the analysis, offering innovative strategies to use news sentiment as a predictive tool for stock market trends.

## Dataset Overview
### Financial News and Stock Price Integration Dataset
The FNSPID (Financial News and Stock Price Integration Dataset) is designed to enhance stock market predictions by combining quantitative and qualitative data. The dataset includes:
- **headline**: Article release headline, often including key financial actions like stocks hitting highs, price target changes, or company earnings.
- **url**: The direct link to the full news article.
- **publisher**: Author or creator of the article.
- **date**: The publication date and time, including timezone information (UTC-4 timezone).
- **stock**: Stock ticker symbol (a unique series of letters assigned to a publicly traded company, e.g., AAPL for Apple).

## Todo
### Exploratory Data Analysis (EDA)
#### Descriptive Statistics
- Obtain basic statistics for textual lengths (e.g., headline length).
- Count the number of articles per publisher to identify the most active publishers.
- Analyze publication dates to identify trends over time, such as increased news frequency on specific days or during events.

#### Text Analysis (Sentiment Analysis & Topic Modeling)
- Perform sentiment analysis on headlines to gauge sentiment (positive, negative, neutral).
- Use natural language processing to identify common keywords or phrases and extract topics or significant events (e.g., "FDA approval", "price target").

#### Time Series Analysis
- Analyze how publication frequency varies over time and identify spikes related to specific market events.
- Analyze publishing times to determine if there’s a specific time when most news is released, which could be valuable for traders and automated trading systems.

#### Publisher Analysis
- Identify which publishers contribute the most to the news feed and examine differences in the types of news they report.
- If email addresses are used as publisher names, identify unique domains to see if certain organizations contribute more frequently.

#### Correlation Analysis
- Perform correlation analysis between different sentiment scores and key financial metrics.
