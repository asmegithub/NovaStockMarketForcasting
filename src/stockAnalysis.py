import re

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob

class StockMarketAnalysis:
    def __init__(self, file_path):
        self.file_path = file_path
        # read .csv file
    def load_data(self):
        self.data = pd.read_csv(self.file_path)
        return self.data
        # check data quality
    def check_data_quality(self):
        # Check for missing values
        missing_values = self.data.isnull().sum()
        print("Missing values:")
        print(missing_values)

        # Check for duplicates
        duplicates = self.data.duplicated().sum()
        print("\n Number of duplicates:", duplicates)

        # Check data types
        data_types = self.data.dtypes
        print("\n Data types:")
        print(data_types)
    
     # Drops the 'Unnamed' column from a pandas DataFrame
    def drop_unnamed_column(self):
        
        self.data = self.data.drop(['Unnamed: 0'], axis=1)
        return self.data
    
        # Descriptive Statistics Analysis
    def headline_length_stats(self):
        
        self.data['headline_length'] = self.data['headline'].str.len()
        print("Headline Length Statistics:")
        return self.data['headline_length'].describe()
    
      # Number of Articles in each publisher 
    def article_per_publisher(self):
        publisherCounts = self.data['publisher'].value_counts().to_frame().reset_index()
        publisherCounts.columns = ['Publisher', 'Article Count']
        print("Article Counts per Publisher:")
        return publisherCounts
    
    def plot_article_per_publisher(self):
        publisherCounts = self.article_per_publisher().nlargest(5, 'Article Count')
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Publisher', y='Article Count', data=publisherCounts)
        plt.xticks(rotation=90)
        plt.xlabel('Publisher')
        plt.ylabel('Number of Articles')
        plt.title('Number of Articles for each Publisher')
        plt.show()
    
    def analyze_publication_dates(self):
        """Analyze publication dates to identify trends over time."""
        self.data['date'] = pd.to_datetime(self.data['date'], errors='coerce',utc=True)
        self.data['year'] = self.data['date'].dt.year
        self.data['month'] = self.data['date'].dt.month
        self.data['day_of_week'] = self.data['date'].dt.day_name()

        # Articles over time by year and month
        articlesOverTime = self.data.groupby(['year', 'month']).size().reset_index(name='article_count')

        # Articles by day of the week
        articlesByDay = self.data['day_of_week'].value_counts()

        return articlesOverTime, articlesByDay
    
    def plot_article_trends(self, articles_over_time, articles_by_day):
        """Plot trends of article counts over time and by day of the week."""
        # Plotting article counts over time
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=articles_over_time, x='month', y='article_count', hue='year', marker='o')
        plt.title('Article Count Time')
        plt.xlabel('Month')
        plt.ylabel('Number of Articles')
        plt.show()

        # Plotting article counts by day of the week
        plt.figure(figsize=(10, 5))
        sns.barplot(x=articles_by_day.index, y=articles_by_day.values)
        plt.title('Article Count by Day of the Week')
        plt.xlabel('Day of the Week')
        plt.ylabel('Number of Articles')
        plt.show()

    # Text Analyis
    def text_preprocess(self):
        # Convert to lowercase and remove non-alphabetic characters
        self.data['cleaned_headline']=self.data['headline'].str.lower().str.replace(r'[^a-zA-Z\s]', '', regex=True) 
        # Remove leading and trailing whitespace
        self.data['cleaned_headline']=self.data['cleaned_headline'].str.strip() 
        # remove stop words
        stop_words = set(stopwords.words('english'))
        self.data['cleaned_headline'] = self.data['cleaned_headline'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
        return self.data
    
    def get_sentiment(self):
        # First preprocess the text
        self.text_preprocess()
        # Calculate the sentiment polarity of each headline
        self.data['polarity'] = self.data['cleaned_headline'].apply(lambda x: TextBlob(x).sentiment.polarity)
        # Categorize the sentiment based on the polarity score
        self.data['sentiment'] = self.data['polarity'].apply(lambda x: 'positive' if x > 0 else 'Negative' if x < 0 else 'Neutral')
        return self.data
    def plot_sentiment_distribution(self):
        # First preprocess the text
        self.text_preprocess()
        
        # Visualize the sentiment distribution
        sentiment_counts = self.get_sentiment()['sentiment'].value_counts()
        print(sentiment_counts) 
        plt.figure(figsize=(8, 6))
        sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values)
        plt.xticks(rotation=45)  
        plt.xlabel('Sentiment')  
        plt.ylabel('Number of Articles') 
        plt.title('Sentiment Distribution') 
        plt.show()  
    def word_frequency(self):
        # First preprocess the text
        self.text_preprocess()
        # Concatenate all headlines in a single string
        all_text = ' '.join(self.data['cleaned_headline']) 
        # Count the frequency of each word
        word_freq = pd.Series(all_text.split()).value_counts() 
        # Plot the top 20 most frequent words
        word_freq[:20].plot(kind='bar', figsize=(12, 6))
        plt.xticks(rotation=45)
        plt.xlabel('Words') 
        plt.ylabel('Frequency')
        plt.title('Word Frequency Distribution')
        plt.show()        
    
    # Keyword Extraction using TF-IDF
    def extract_keywords(self, n_keywords=5):
        # Initialize TF-IDF Vectorizer
        self.text_preprocess()
        vectorizer = TfidfVectorizer(max_features=n_keywords)
        tfidf_matrix = vectorizer.fit_transform(self.data['cleaned_headline'])
        
        # Extract keywords
        keywords = vectorizer.get_feature_names_out()
        return keywords
    
       # Topic Modeling
    def perform_topic_modeling(self, n_topics=2):
        self.text_preprocess()
        # Initialize TF-IDF Vectorizer
        tfvectorizer = TfidfVectorizer(stop_words='english')
        tfidfMatrix = tfvectorizer.fit_transform(self.data['cleaned_headline'])
        
        # Perform LDA
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=0)
        lda.fit(tfidfMatrix)
        
        # Display Topics
        words = tfvectorizer.get_feature_names_out()
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            topicKeywords = [words[i] for i in topic.argsort()[:-n_topics - 1:-1]]
            topics.append(f"Topic {topic_idx+1}: " + ", ".join(topicKeywords))
        
        return topics
    
    # Time Series Analysis
    def analyze_time_series(self):
        """ Perform time series analysis to understand publication frequency related to market events."""
        # Convert 'date' to datetime
        if 'date' not in self.data or self.data['date'].dtype != 'datetime64[ns]':
            self.data['date'] = pd.to_datetime(self.data['date'], errors='coerce')

        # Group by date to see frequency over time
        publicationFrequency = self.data['date'].value_counts().sort_index()

        # Extract hour from date to analyze publishing times
        self.data['hour'] = self.data['date'].dt.hour
        publishingTimes = self.data['hour'].value_counts().sort_index()

        return publicationFrequency, publishingTimes

    def plot_time_series_trends(self, publicationFrequency, publishingTimes):
        """
        Plot the time series analysis results, including publication frequency and publishing times.
        publicationFrequency: pd.Series of publication counts by date.
        publishingTimes: pd.Series of publication counts by hour.
        """
        # Plot publication frequency over time
        plt.figure(figsize=(12, 6))
        plt.plot(publicationFrequency.index, publicationFrequency.values, marker='o', label="Publication Frequency")
        plt.title('Publication Frequency Over Time')
        plt.xlabel('Date')
        plt.ylabel('Number of Articles')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend()
        plt.show()

        # Plot publishing times by hour
        plt.figure(figsize=(12, 6))
        sns.barplot(x=publishingTimes.index, y=publishingTimes.values, palette='viridis')
        plt.title('Publication Count by Hour of the Day')
        plt.xlabel('Hour of the Day')
        plt.ylabel('Number of Articles')
        plt.grid(True)
        plt.show()

    # Publisher Analysis
    def _extract_domain_from_email(self, email):
        """
        Extracts the domain from an email address.

        param email: str, email address
        return: str, domain extracted from the email
        """
        match = re.search(r'@([\w.-]+)', email)
        return match.group(1) if match else None
    
    def analyze_publishers(self):
        """
        Analyze the publisher data to determine the top publishers (emails) and domains,
        as well as publishers without domains.

        return: tuple (pd.Series, pd.Series, pd.Series)
                 publishers_with_domain: Frequency count of publishers with domains (emails).
                 publishers_without_domain: Frequency count of publishers without domains.
                 publisher_domains: Frequency count of domains from publishers with emails.
        """
        # Extract domains from publishers
        self.data['domain'] = self.data['publisher'].apply(self._extract_domain_from_email)

        # Separate publishers with and without domains
        publishers_with_domain = self.data.dropna(subset=['domain'])
        publishers_without_domain = self.data[self.data['domain'].isna()]

        # Count frequency of publishers with domains
        top_publishers_with_domain = publishers_with_domain['publisher'].value_counts()

        # Count frequency of publishers without domains
        top_publishers = publishers_without_domain['publisher'].value_counts()

        # Count frequency of domains
        publisher_domains = publishers_with_domain['domain'].value_counts()

        return top_publishers_with_domain, top_publishers, publisher_domains

    def plot_publisher_analysis(self, publishers_with_domain, publishers_without_domain, publisher_domains):
        """
        Plot analysis of publishers with and without domains, and their respective counts.

        param publishers_with_domain: publishers with domains and their article counts.
        param publishers_without_domain: publishers without domains and their article counts.
        param publisher_domains: The domains extracted from publishers column.
        """
        # Plot publishers with domains
        plt.figure(figsize=(12, 6))
        sns.barplot(x=publishers_with_domain.index[:5], y=publishers_with_domain.values[:5], palette='coolwarm')
        plt.title('Top 5 Publishers with domain by Article Count')
        plt.xlabel('Publisher with Domain')
        plt.ylabel('Number of Articles')
        plt.xticks(rotation=45)
        plt.show()

        # Plot publishers without domains
        plt.figure(figsize=(10, 6))
        sns.barplot(x=publishers_without_domain.index[:5], y=publishers_without_domain.values[:5], palette='coolwarm')
        plt.title('Top 5 Publishers by Article Count')
        plt.xlabel('Publisher without Domain')
        plt.ylabel('Number of Articles')
        plt.xticks(rotation=45)
        plt.show()

        # Plot top domains
        plt.figure(figsize=(10, 6))
        sns.barplot(x=publisher_domains.index[:5], y=publisher_domains.values[:5], palette='coolwarm')
        plt.title('Top 5 Publisher Domains by Article Count')
        plt.xlabel('Domain')
        plt.ylabel('Number of Articles')
        plt.xticks(rotation=45)
        plt.show()