# News Information and data article 
from newspaper import Article, Config
from gnews import GNews

# Data Analysis and Profiling
import pandas as pd
from ydata_profiling import ProfileReport
from st_aggrid import AgGrid

# Streamlit for Building the Dashboard
import streamlit as st
from streamlit_pandas_profiling import st_profile_report

# Language Detection
from langdetect import detect

# NLP and Text Processing
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from deep_translator import GoogleTranslator
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup

# Sentiment Analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# URL Parsing
from urllib.parse import urlparse

# Data Visualization
import plotly.express as px
import matplotlib.pyplot as plt

# Word Cloud Generation
from wordcloud import WordCloud

# Other Libraries
import torch
import requests
import subprocess
import logging
import json
import re
import os

# NLTK Data Download
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

## ............................................... ##
# Set page configuration (Call this once and make changes as needed)
st.set_page_config(page_title='News Scrapping',  layout='wide', page_icon=':newspaper:')

with st.container():
    # Initialize Streamlit app
    st.title('News Article Scrapping')
    st.write("Created by Bayhaqy")

## ............................................... ##
# Set up logging
logging.basicConfig(filename='news_processing.log', level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

## ............................................... ##
# Function for get model and tokenize
@st.cache_resource
def get_models_and_tokenizers():
    model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    #model.eval()

    return model, tokenizer

# Function for sentiment analysis
@st.cache_resource
def analyze_sentiment_distilbert(text, _model, _tokenizer):
    try:
        tokens_info = _tokenizer(text, truncation=True, return_tensors="pt")
        with torch.no_grad():
            raw_predictions = _model(**tokens_info).logits

        predicted_class_id = raw_predictions.argmax().item()
        predict = _model.config.id2label[predicted_class_id]

        softmaxed = int(torch.nn.functional.softmax(raw_predictions[0], dim=0)[1] * 100)
        if (softmaxed > 70):
            status = 'Not trust'
        elif (softmaxed > 40):
            status = 'Not sure'
        else:
            status = 'Trust'
        return status, predict

    except Exception as e:
        logging.error(f"Sentiment analysis error: {str(e)}")
        return 'N/A', 'N/A'

# Function for sentiment analysis using VADER
@st.cache_data
def analyze_sentiment_vader(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(text)
    compound_score = sentiment['compound']
    if compound_score >= 0.05:
        return 'Positive'
    elif compound_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Function for sentiment analysis using TextBlob
@st.cache_data
def analyze_sentiment_textblob(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

## ............................................... ##
# Function to process an article
@st.cache_data
def process_article(url, _config):
    try:
        article = Article(url=url, config=_config)
        article.download()
        article.parse()

        # Check if publish_date is not None before further processing
        if article.publish_date is None:
            return None  # Skip processing and return None

        # Check if text is not None before further processing
        if len(article.text) <= 5:
            return None  # Skip processing and return None

        # Get the article data if publish_date is not not None
        text = article.text
        url = article.canonical_link
        source_url = urlparse(url).netloc

        title = article.title
        authors = article.authors
        #publish_date = article.publish_date.strftime('%Y-%m-%d %H:%M:%S%z')
        publish_date = article.publish_date.strftime('%Y-%m-%d %H:%M')

        article.nlp()
        keywords = article.meta_keywords
        summary = article.summary

        language = detect(title)

        return publish_date, language, url, source_url, title, authors, keywords, text, summary

    except Exception as e:
        logging.error(f"Article processing error: {str(e)}")
        return None  # Skip processing and return None

# Function for translation
@st.cache_data
def translate_text(text, source='auto', target='en'):
    try:
        if source != target:
            text = GoogleTranslator(source=source, target=target).translate(text)
        return text

    except Exception as e:
        logging.error(f"Translation error: {str(e)}")
        return text

## ............................................... ##
# Function to preprocess the data
@st.cache_data
def preprocessing_data(df):
    # Remove duplicates
    df = df.drop_duplicates(subset='Translation')

    # Reset the index to add the date column
    df.reset_index(inplace=True,drop=True)

    # Function to clean and preprocess text
    def clean_text(text):
        # Remove URLs
        text = re.sub(r'http\S+', '', text)

        # Convert to lowercase
        text = text.lower()

        # Remove non-alphanumeric characters
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Tokenize text
        words = nltk.word_tokenize(text)

        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]

        # Lemmatize words
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]

        return ' '.join(words)

    # Apply the clean_text function to the "Translation" column
    df['Cleaned Translation'] = df['Translation'].apply(clean_text)

    return df
    
## ............................................... ##
# Function to create a Word Cloud
@st.cache_data
def create_wordcloud(df):
    # Combine all text
    text = ' '.join(df['Cleaned Translation'])

    # Create a Word Cloud
    wordcloud = WordCloud(width=700, height=400, max_words=80).generate(text)

    # Convert the word cloud to an image
    wordcloud_image = wordcloud.to_image()

    # Display the Word Cloud using st.image
    st.image(wordcloud_image, use_column_width=True)

## ............................................... ##
with st.container():
    # Input search parameters
    search_term = st.text_input('Enter a search term :', 'Indonesia')

    col1, col2, col3 = st.columns(3)

    with col1:
        period = st.text_input('Enter a news period :', '7d')
        max_results = st.number_input('Maximum number of results :', min_value=1, value=10)
    with col2:
        country = st.text_input('Country :', 'Indonesia')
        language = st.text_input('Language :', 'indonesian')
    with col3:  
        start_date = st.date_input('Start Date :', pd.to_datetime('2023-01-01'))
        end_date = st.date_input('End Date :', pd.to_datetime('2023-12-01'))

## ............................................... ##
with st.container():
    col1, col2 = st.columns(2)

    with col1:
        # Checkbox options for different processing steps
        include_translation = st.checkbox("Include Translation", value=True)
        include_sentiment_analysis = st.checkbox("Include Sentiment Analysis", value=True)
    with col2:
        include_sentiment_vader = st.checkbox("Include VADER Sentiment Analysis", value=True)
        include_sentiment_textblob = st.checkbox("Include TextBlob Sentiment Analysis", value=True)

## ............................................... ##
# Create a variable to track whether the data has been processed
data_processed = False

## ............................................... ##
# Create a custom configuration
config = Config()
config.number_threads = 500
config.request_timeout = 10

## ............................................... ##
# Initialize the DataFrame
df = pd.DataFrame(columns=['Publish_Date', 'Language', 'URL', 'Source_Url', 'Title', 'Authors', 'Keywords', 'Text', 'Summary']) 

# Initialize your model and tokenizer
model, tokenizer = get_models_and_tokenizers()

## ............................................... ##
with st.container():
    # Fetch news and process articles
    if st.button('Fetch and Process News'): 
        # Your news retrieval code
        google_news = GNews()
        google_news.period = period  # News from last 7 days
        google_news.max_results = max_results # number of responses across a keyword
        google_news.country = country  # News from a specific country
        google_news.language = language  # News in a specific language
        #google_news.exclude_websites = ['yahoo.com', 'cnn.com']  # Exclude news from specific website i.e Yahoo.com and CNN.com
        google_news.start_date = (start_date.year, start_date.month, start_date.day) # Search from 1st Jan 2023
        google_news.end_date = (end_date.year, end_date.month, end_date.day) # Search until 1st Dec 2023
        
        news = google_news.get_news(search_term)
        
        ## ............................................... ##,
        # Progress bar for fetching and processing news
        progress_bar = st.progress(0)
        total_news = len(news)
        
        # Your news retrieval code (assuming 'news' is a list of article URLs)
        #for x in news:
        for idx, x in enumerate(news):
            result = process_article(x['url'], _config=config)
            if result is not None:
                publish_date, language, url, source_url, title, authors, keywords, text, summary = result

                # Insert to dataframe
                temp_df = pd.DataFrame({'Publish_Date': [publish_date], 'Language': [language], 'URL': [url], 'Source_Url': [source_url], 'Title': [title], 'Authors': [authors], 'Keywords': [keywords],
                                      'Text': [text], 'Summary': [summary]})
                df = pd.concat([df, temp_df], ignore_index=True)

                # Convert 'Publish_Date' to DatetimeIndex
                df['Publish_Date'] = pd.to_datetime(df['Publish_Date'])
            
            # Update the progress bar
            progress = (idx + 1) / total_news
            progress_bar.progress(progress)
        
        # Conditionally apply translation function to the 'Translation' column
        if include_translation:
            df['Translation'] = df.apply(lambda row: translate_text((row['Title'] + ' | ' + row['Summary']), source=row['Language'], target='en'), axis=1)
            
            # Preprocessing Data
            df = preprocessing_data(df)
        
        # Conditionally apply sentiment analysis function to the 'Translation' column
        if include_sentiment_analysis:
            df[['Fake Check', 'Sentiment Distilbert']] = df['Translation'].apply(lambda text: pd.Series(analyze_sentiment_distilbert(text, model, tokenizer)))
          
        
        # Conditionally apply VADER sentiment analysis to the 'Translation' column
        if include_sentiment_vader:
            df['Sentiment VADER'] = df['Translation'].apply(analyze_sentiment_vader)
        
        # Conditionally apply TextBlob sentiment analysis to the 'Translation' column
        if include_sentiment_textblob:
            df['Sentiment TextBlob'] = df['Translation'].apply(analyze_sentiment_textblob)
        
        # Set data_processed to True when the data has been successfully processed
        data_processed = True
        
    ## ............................................... ##
    # Add a button to download the data as a CSV file
    if data_processed:
        st.markdown("### Download Processed Data as CSV")
        st.write("Click the button below to download the processed data as a CSV file.")
        
        # Create a downloadable link
        csv_data = df.to_csv(index=False).encode()
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name="processed_data.csv",
        )

    ## ............................................... ##
    with st.expander("See for Table"):
        # Display processed data
        if data_processed:
            AgGrid(df, height=400)
    
    ## ............................................... ##
    # Display processed data
    with st.expander("See for Exploratory Data Analysis"):
        if data_processed:
            col1, col2 = st.columns(2)
            with col1:
                ## ............................................... ##
                # Create a DataFrame to count the number of tweets by Fake Check
                FakeCheck_counts = df['Fake Check'].value_counts().reset_index()
                FakeCheck_counts.columns = ['Fake Check', 'News Count']
                fig = px.bar(FakeCheck_counts, x='Fake Check', y='News Count', text='News Count', title='Total News by Fake Check')
                st.plotly_chart(fig, use_container_width=True, use_container_height=True, width=700, height=400)
        
                ## ............................................... ##
                # Create wordcloud
                try:
                    st.write('WordCloud for News')
                    create_wordcloud(df)
                except Exception as e:
                    logging.error(f" Column Translation Not Available : {str(e)}")
        
                ## ............................................... ##

            with col2:
                ## ............................................... ##
                # Create a DataFrame to count the number of News by language
                language_counts = df['Language'].value_counts().reset_index()
                language_counts.columns = ['Language', 'News Count']
                fig = px.bar(language_counts, x='Language', y='News Count', text='News Count', title='Total News by Language')
                st.plotly_chart(fig, use_container_width=True, use_container_height=True, width=700, height=400)
                
                ## ............................................... ##
                # Group by Sentiment columns and get the count
                try:
                    sentiment_counts = df[['Sentiment Distilbert', 'Sentiment VADER', 'Sentiment TextBlob']].apply(lambda x: x.value_counts()).T
                    sentiment_counts = sentiment_counts.reset_index()
                    sentiment_counts = pd.melt(sentiment_counts, id_vars='index', var_name='Sentiment', value_name='Count')
                    fig = px.bar(sentiment_counts, x='Sentiment', y='Count', color='index', barmode='group', title='Total News per Sentiment')
                    st.plotly_chart(fig, use_container_width=True, use_container_height=True, width=700, height=400)
        
                except Exception as e:
                    logging.error(f" Column Sentiment Not Available : {str(e)}")
            
                ## ............................................... ##
        
    with st.expander("See for Analysis with ydata-profiling"):
        ## ............................................... ##
        # Display processed data
        if data_processed:
            pr = ProfileReport(df)
            st_profile_report(pr)