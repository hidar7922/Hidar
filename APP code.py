# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 08:58:17 2024

@author: haider
"""

import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
from transformers import pipeline
import evaluate
import re

# Load Summarization and Sentiment Models
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
rouge = evaluate.load("rouge")

# Function to fetch articles from a URL
def fetch_articles(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        articles = []
        for link in soup.find_all("a", href=True):
            href = link["href"]
            if "/news/" in href and href.startswith("http"):
                articles.append(href)
        return list(set(articles))  # Return unique links
    else:
        st.error(f"Failed to fetch articles. HTTP Status Code: {response.status_code}")
        return []

# Function to fetch and clean content of a single article
def fetch_article_content(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        paragraphs = soup.find_all("p")
        content = " ".join([para.get_text(strip=True) for para in paragraphs])
        return re.sub(r"\s+", " ", content).strip()  # Clean extra whitespaces
    else:
        return None

# Function to preprocess text
def preprocess_text(text):
    text = re.sub(r"\s+", " ", text)  # Remove extra spaces
    text = re.sub(r"[^a-zA-Z0-9.,!? ]", "", text)  # Remove special characters
    return text.strip()

# Function to summarize content using BART
def summarize_content(content):
    summary = summarizer(content, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, truncation=True)
    return summary[0]['summary_text']

# Function to analyze sentiment
def analyze_sentiment(text):
    sentiment = sentiment_analyzer(text)
    return sentiment[0]  # Return first result

# Streamlit App
st.title("Enhanced News Article Analysis Application")
st.sidebar.header("Settings")

# Input: News Website URL
news_url = st.text_input("Enter News Website URL", "https://www.thestar.com.my/")

# Button to Fetch and Process Articles
if st.button("Fetch and Analyze Articles"):
    with st.spinner("Fetching articles..."):
        article_links = fetch_articles(news_url)
    
    if article_links:
        st.success(f"Found {len(article_links)} articles.")
        all_data = []

        for link in article_links[:5]:  # Limit to 5 articles for demo
            with st.spinner(f"Processing article: {link}"):
                content = fetch_article_content(link)
                if content:
                    preprocessed_content = preprocess_text(content)
                    summary = summarize_content(preprocessed_content)
                    sentiment = analyze_sentiment(preprocessed_content)
                    
                    all_data.append({
                        "URL": link,
                        "Content": preprocessed_content,
                        "Summary": summary,
                        "Sentiment": sentiment['label'],
                        "Sentiment Score": sentiment['score'],
                    })
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        
        st.subheader("Analysis Results")
        st.write(df)

        # Download Option
        st.download_button(
            label="Download Results",
            data=df.to_csv(index=False),
            file_name="enhanced_news_analysis.csv",
            mime="text/csv",
        )
    else:
        st.error("No articles found. Please check the URL.")
