# financial_dashboard.py

import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from datetime import date

# ------------------- Streamlit Page Setup -------------------
st.set_page_config(page_title="Financial Dashboard", layout="wide")
st.title("📊 Financial Dashboard")
st.markdown("A clean and interactive dashboard for stock analysis & portfolio performance.")

# ------------------- Sidebar Inputs -------------------
st.sidebar.header("⚙️ User Settings")
tickers_input = st.sidebar.text_input("Enter Stock Tickers (comma separated)", "AAPL,MSFT,GOOGL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime(date.today()))
portfolio_allocations = st.sidebar.text_area(
    "Enter Portfolio Allocations (ticker:percentage)",
    "AAPL:40,MSFT:40,GOOGL:20"
)

st.sidebar.markdown("---")
st.sidebar.info("💡 Tip: Enter tickers like `AAPL, TSLA, AMZN` and allocations like `AAPL:50,TSLA:30,AMZN:20`.")

tickers = [ticker.strip().upper() for ticker in tickers_input.split(",")]

# Parse allocations
alloc_dict = {}
for item in portfolio_allocations.split(","):
    try:
        t, p = item.split(":")
        alloc_dict[t.strip().upper()] = float(p.strip()) / 100
    except:
        pass

# ------------------- Functions -------------------
def fetch_data(ticker):
    data = yf.download(ticker, start=start_date, end=end_date)

    # If data has multi-level columns, flatten it
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Moving averages
    data['50_SMA'] = data['Close'].rolling(50).mean()
    data['200_SMA'] = data['Close'].rolling(200).mean()
    data['20_EMA'] = data['Close'].ewm(span=20, adjust=False).mean()

    # Bollinger Bands (force Close to Series)
    close_series = data['Close'].astype(float)
    rolling_std = close_series.rolling(20).std()
    data['Upper_BB'] = data['20_EMA'] + 2 * rolling_std
    data['Lower_BB'] = data['20_EMA'] - 2 * rolling_std

    # RSI
    delta = close_series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    RS = gain / loss
    data['RSI'] = 100 - (100 / (1 + RS))

    return data


def candlestick_chart(df, ticker):
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                                         open=df['Open'], high=df['High'],
                                         low=df['Low'], close=df['Close'],
                                         name='Candlestick')])
    fig.add_trace(go.Scatter(x=df.index, y=df['50_SMA'], mode='lines', name='50-SMA'))
    fig.add_trace(go.Scatter(x=df.index, y=df['200_SMA'], mode='lines', name='200-SMA'))
    fig.add_trace(go.Scatter(x=df.index, y=df['Upper_BB'], line=dict(color='gray', dash='dot'), name='Upper BB'))
    fig.add_trace(go.Scatter(x=df.index, y=df['Lower_BB'], line=dict(color='gray', dash='dot'), name='Lower BB'))
    fig.update_layout(title=f"{ticker} Price Chart with Indicators",
                      xaxis_title="Date", yaxis_title="Price ($)", height=600)
    return fig

def rsi_chart(df, ticker):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI'))
    fig.add_hline(y=70, line_dash="dash", line_color="red")
    fig.add_hline(y=30, line_dash="dash", line_color="green")
    fig.update_layout(title=f"{ticker} RSI", xaxis_title="Date", yaxis_title="RSI", height=300)
    return fig

def portfolio_performance(alloc_dict, start, end):
    prices = pd.DataFrame()
    for t in alloc_dict.keys():
        prices[t] = yf.download(t, start=start, end=end)['Close']
    prices.fillna(method='ffill', inplace=True)
    returns = prices.pct_change().dropna()
    weighted_returns = returns.mul([alloc_dict[t] for t in alloc_dict.keys()], axis=1)
    portfolio_return = weighted_returns.sum(axis=1)
    cumulative = (1 + portfolio_return).cumprod()

    metrics = {
        "Average Daily Return": portfolio_return.mean(),
        "Volatility (Std Dev)": portfolio_return.std(),
        "Sharpe Ratio (assume RF=0)": portfolio_return.mean() / portfolio_return.std() if portfolio_return.std() != 0 else 0
    }
    return cumulative, metrics

# ------------------- Layout -------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📈 Stock Analysis")
    for ticker in tickers:
        df = fetch_data(ticker)
        st.subheader(f"Ticker: {ticker}")
        st.plotly_chart(candlestick_chart(df, ticker), use_container_width=True)
        st.plotly_chart(rsi_chart(df, ticker), use_container_width=True)

with col2:
    st.header("💼 Portfolio Overview")
    if len(alloc_dict) > 0:
        cumulative, metrics = portfolio_performance(alloc_dict, start_date, end_date)
        
        # Growth chart
        fig_port = go.Figure()
        fig_port.add_trace(go.Scatter(x=cumulative.index, y=cumulative, mode='lines', name='Portfolio Growth'))
        fig_port.update_layout(title="Cumulative Growth", xaxis_title="Date", yaxis_title="Growth ($)")
        st.plotly_chart(fig_port, use_container_width=True)

        # Allocation pie
        fig_pie = go.Figure(go.Pie(labels=list(alloc_dict.keys()), values=list(alloc_dict.values()), hole=0.3))
        fig_pie.update_layout(title="Portfolio Allocation")
        st.plotly_chart(fig_pie, use_container_width=True)

        # Metrics
        st.subheader("📊 Key Metrics")
        for k, v in metrics.items():
            st.write(f"**{k}:** {v:.4f}")
