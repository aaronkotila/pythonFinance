import streamlit as st
import pandas as pd
import datetime as dt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from bs4 import BeautifulSoup
import requests
import plotly.express as px

st.set_page_config(page_title="Stock Sentiment Analyzer", layout="wide")
st.title("Financial Sentiment Analyzer")

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

@st.cache_data
def get_news(ticker):
    url = f"https://finviz.com/quote.ashx?t={ticker}&p=d" 
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    
    try:
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        news_table = soup.find(id='news-table')
        
        if not news_table:
            return None

        parsed_news = []
        
        for row in news_table.find_all('tr'):
            if row.a is None:
                continue

            title = row.a.get_text()
            date_data = row.td.text.split()
            
            if len(date_data) == 1:
                time = date_data[0]
            else:
                date = date_data[0]
                time = date_data[1]
                
            parsed_news.append([date, time, title])
            
        df = pd.DataFrame(parsed_news, columns=['Date', 'Time', 'Headline'])
        return df

    except Exception as e:
        st.error(f"Error scraping data: {e}")
        return None

def analyze_sentiment(df):
    vader = SentimentIntensityAnalyzer()
    scores = df['Headline'].apply(vader.polarity_scores).tolist()
    scores_df = pd.DataFrame(scores)
    result_df = pd.concat([df, scores_df], axis=1)
    return result_df

ticker = st.text_input("Enter Stock Ticker", "MU").upper()

if ticker:
    st.write(f"Fetching news for **${ticker}**...")
    raw_news = get_news(ticker)
    
    if raw_news is not None and not raw_news.empty:
        scored_news = analyze_sentiment(raw_news)
        
        scored_news['Sentiment'] = scored_news['compound'].apply(
            lambda x: 'Positive' if x > 0.05 else ('Negative' if x < -0.05 else 'Neutral')
        )

        avg_sentiment = scored_news['compound'].mean()
        
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Average Sentiment Score", f"{avg_sentiment:.2f}", 
                      help="Range: -1 (Neg) to +1 (Pos)")
            
            if avg_sentiment > 0.05:
                st.success("Overall Sentiment: Bullish")
            elif avg_sentiment < -0.05:
                st.error("Overall Sentiment: Bearish")
            else:
                st.warning("Overall Sentiment: Neutral")

        with col2:
            sentiment_counts = scored_news['Sentiment'].value_counts()
            fig = px.pie(
                values=sentiment_counts.values, 
                names=sentiment_counts.index, 
                title=f"News Sentiment Distribution for {ticker}",
                color=sentiment_counts.index,
                color_discrete_map={'Positive':'green', 'Negative':'red', 'Neutral':'gray'}
            )
            st.plotly_chart(fig, width="stretch")

        st.subheader("Latest Headlines & Scores")
        st.dataframe(scored_news[['Date', 'Time', 'Headline', 'Sentiment', 'compound']], width="stretch")
        
    else:
        st.warning("No news found. The ticker might be wrong or Finviz is blocking requests.")
        
