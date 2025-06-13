import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
from sklearn.linear_model import LinearRegression
import numpy as np
from matplotlib.ticker import MaxNLocator

# === Setup ===
st.set_page_config(page_title="Stock Sentiment Dashboard", layout="wide")
st.markdown("<style>body {background-color: #e6f5e6;}</style>", unsafe_allow_html=True)
st.markdown("<h1 style='color:#2e7d32;'>📊 Stock News Sentiment Dashboard</h1>", unsafe_allow_html=True)

# === API Keys ===
NEWS_API_KEY = "enter you news api key "
STOCK_API_KEY = "enter you stock api key "

# === Cached Models and Data ===
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

@st.cache_resource
def load_summary_model():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

@st.cache_data
def get_stock_symbol(company_name):
    try:
        url = "https://www.alphavantage.co/query"
        params = {"function": "SYMBOL_SEARCH", "keywords": company_name, "apikey": STOCK_API_KEY}
        response = requests.get(url, params=params)
        if response.status_code == 200:
            matches = response.json().get("bestMatches", [])
            if matches:
                return matches[0]["1. symbol"]
    except:
        pass
    return None

@st.cache_data
def fetch_news(api_key, keyword):
    try:
        url = f'https://newsdata.io/api/1/news?apikey={api_key}&q={keyword}&language=en'
        response = requests.get(url)
        articles = []
        if response.status_code == 200:
            data = response.json()
            for article in data.get("results", [])[:10]:
                title = article.get("title", "")
                link = article.get("link", "")
                text = article.get("description", title)
                articles.append({"title": title, "link": link, "text": text})
        return articles
    except:
        return []

@st.cache_data
@st.cache_data
def summarize_articles(articles):
    summarizer = load_summary_model()
    summaries = []
    for art in articles:
        text = art.get("text", "")
        if not isinstance(text, str) or not text.strip():
            summaries.append({**art, "summary": "No content available."})
            continue
        word_count = len(text.split())
        if word_count > 50:
            try:
                input_text = text if len(text) < 512 else text[:512]
                summary = summarizer(input_text, max_length=60, min_length=15, do_sample=False)[0]['summary_text']
            except Exception:
                summary = text[:150] + "..."
        else:
            summary = text
        summaries.append({**art, "summary": summary})
    return summaries

@st.cache_data
def analyze_articles(articles):
    model = load_sentiment_model()
    label_map = {'LABEL_0': 'Negative', 'LABEL_1': 'Neutral', 'LABEL_2': 'Positive'}
    results = []
    for art in articles:
        try:
            result = model(art["text"][:512])[0]
            sentiment = label_map.get(result["label"], "Unknown")
        except Exception:
            sentiment = "Neutral"
        results.append({"Title": art["title"], "Sentiment": sentiment, "Link": art["link"], "summary": art.get("summary", "")})
    return pd.DataFrame(results)

@st.cache_data
@st.cache_data
def fetch_stock_data(symbol, interval="15min"):
    if not symbol:
        st.warning("⚠️ No stock symbol provided.")
        return pd.DataFrame()
    try:
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": interval,
            "apikey": STOCK_API_KEY,
            "outputsize": "compact"
        }
        response = requests.get(url, params=params)
        data = response.json()
        
        # Debug print
        if "Note" in data:
            st.error("🚫 API call frequency limit reached. Please wait a minute and try again.")
            return pd.DataFrame()
        if "Error Message" in data:
            st.error(f"🚫 Error fetching stock data: {data['Error Message']}")
            return pd.DataFrame()

        key = f"Time Series ({interval})"
        if key in data:
            df = pd.DataFrame.from_dict(data[key], orient="index")
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)
            df = df.rename(columns={
                '1. open': 'Open',
                '2. high': 'High',
                '3. low': 'Low',
                '4. close': 'Close',
                '5. volume': 'Volume'
            })
            return df
        else:
            st.warning("⚠️ Unexpected data format or no stock data found.")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"⚠️ Exception while fetching stock data: {e}")
        return pd.DataFrame()
def predict_next_close(df, days_ahead=1):
    if len(df) < 10:
        return None, None
    try:
        df = df.copy()
        df['timestamp'] = df.index.map(pd.Timestamp.toordinal)
        X = df['timestamp'].values.reshape(-1, 1)
        y = df['Close'].astype(float).values
        model = LinearRegression().fit(X, y)
        next_day = X[-1][0] + days_ahead
        predicted_price = model.predict([[next_day]])
        return predicted_price[0], model.score(X, y)
    except:
        return None, None

# === Charts ===
def plot_sentiment(df, chart_type):
    sentiment_counts = df['Sentiment'].value_counts()
    colors = ['#66bb6a', '#ff7043', '#78909c']
    explode = [0.05] * len(sentiment_counts)

    fig, ax = plt.subplots()
    if chart_type == "Pie Chart":
        ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%',
               startangle=140, colors=colors, explode=explode, shadow=True)
        ax.axis('equal')
    elif chart_type == "Donut Chart":
        wedges, _, _ = ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%',
                              startangle=140, colors=colors, explode=explode, shadow=True)
        center_circle = plt.Circle((0,0), 0.70, fc='white')
        fig.gca().add_artist(center_circle)
        ax.axis('equal')
    elif chart_type == "Bar Chart":
        sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette=colors, ax=ax)
        ax.set_ylabel("Number of Articles")
        ax.set_title("Sentiment Distribution")
    st.pyplot(fig)

def plot_stock_graph(df):
    if not df.empty:
        fig, ax = plt.subplots()
        ax.plot(df.index, df['Close'].astype(float), color="green", label="Close Price")
        ax.set_title("📉 Stock Closing Prices")
        ax.set_xlabel("Time")
        ax.set_ylabel("Price (USD)")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.grid(True)
        st.pyplot(fig)
    else:
        st.warning("⚠️ No stock data available.")

# === UI ===
page = st.sidebar.radio("📂 Navigate", ["🏠 Dashboard", "📰 News Analysis", "💹 Stock Graph", "ℹ️ About"])
with st.sidebar:
    st.markdown("## 🔍 Input Options")
    company_name = st.text_input("Enter Company Name", value="Tesla")
    chart_choice = st.selectbox("Choose Chart Type", ["Pie Chart", "Bar Chart", "Donut Chart"])

# === Page Logic ===
if page == "🏠 Dashboard":
    st.subheader("📈 Combined Dashboard")
    if company_name:
        with st.spinner("Fetching data..."):
            raw_articles = fetch_news(NEWS_API_KEY, company_name)
            summarized = summarize_articles(raw_articles)
            df_news = analyze_articles(summarized)
            symbol = get_stock_symbol(company_name)
            stock_data = fetch_stock_data(symbol) if symbol else pd.DataFrame()

        st.success(f"✅ Found {len(df_news)} articles for **{company_name}**")
        plot_sentiment(df_news, chart_choice)

        st.subheader("📁 Sentiment-wise News")
        for sentiment in ['Positive', 'Neutral', 'Negative']:
            subset = df_news[df_news['Sentiment'] == sentiment]
            with st.expander(f"📰 {sentiment} ({len(subset)})"):
                for _, row in subset.iterrows():
                    st.markdown(f"### [{row['Title']}]({row['Link']})")
                    st.write(f"📝 **Summary:** {row['summary']}")

        if not stock_data.empty:
            st.subheader(f"📉 Stock Graph: {symbol}")
            plot_stock_graph(stock_data)
            pred_price, score = predict_next_close(stock_data)
            if pred_price:
                st.info(f"📈 **Predicted Closing Price for Next Period**: ${pred_price:.2f} (R²: {score:.2f})")

elif page == "📰 News Analysis":
    st.subheader("📰 Sentiment Categorized News")
    raw_articles = fetch_news(NEWS_API_KEY, company_name)
    summarized = summarize_articles(raw_articles)
    df_news = analyze_articles(summarized)
    plot_sentiment(df_news, chart_choice)

    for sentiment in ['Positive', 'Neutral', 'Negative']:
        subset = df_news[df_news['Sentiment'] == sentiment]
        with st.expander(f"{sentiment} ({len(subset)})"):
            for _, row in subset.iterrows():
                st.markdown(f"### [{row['Title']}]({row['Link']})")
                st.write(f"📝 **Summary:** {row['summary']}")

elif page == "💹 Stock Graph":
    st.subheader("📉 Company Stock Performance")
    symbol = get_stock_symbol(company_name)
    if symbol:
        st.success(f"Symbol: `{symbol}`")
        stock_data = fetch_stock_data(symbol)
        plot_stock_graph(stock_data)
        pred_price, score = predict_next_close(stock_data)
        if pred_price:
            st.info(f"📈 **Predicted Closing Price for Next Period**: ${pred_price:.2f} (R²: {score:.2f})")
    else:
        st.warning("No stock symbol found.")

elif page == "ℹ️ About":
    st.subheader("ℹ️ About This App")
    st.markdown("""
    - 🔍 Analyzes recent news headlines for sentiment and summary.
    - 🤖 Uses **RoBERTa** and **BART** models.
    - 📈 Fetches real-time stock data and predicts next price.
    - 📊 Interactive charts and expandable news display.
    - 💡 Built with ❤️ using Streamlit and Hugging Face.
    """)
