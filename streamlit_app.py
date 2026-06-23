# ============================================
# CODE 1 DASHBOARD - RANDOM FOREST VERSION
# No TensorFlow - Works on Free Tier
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# --- Page Config ---
st.set_page_config(page_title="Crypto Predictor", layout="wide")
st.title("🔮 Crypto Predictor with ML")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# --- Settings ---
BANKROLL = 100.00
MAX_RISK_PER_TRADE = 0.02
MIN_EDGE = 0.05
PREDICT_WINDOW = 5
LOOKBACK_WINDOW = 30

COINS = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD', 'DOGE-USD']

# --- Cached Data ---
@st.cache_data(ttl=60)
def fetch_yahoo_data(symbol):
    """Fetch OHLCV data from Yahoo Finance"""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period='5d', interval='1m')
        if df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        df = df.rename(columns={
            'Datetime': 'time',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        return df
    except:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_coinpaprika_data(coin_id):
    """Fetch data from CoinPaprika"""
    try:
        url = f"https://api.coinpaprika.com/v1/tickers/{coin_id}"
        response = requests.get(url, timeout=5)
        data = response.json()
        quotes = data.get('quotes', {}).get('USD', {})
        return {
            'price': quotes.get('price', 0),
            'percent_change_15m': quotes.get('percent_change_15m', 0),
            'percent_change_24h': quotes.get('percent_change_24h', 0),
            'volume_24h': quotes.get('volume_24h', 0)
        }
    except:
        return {'price': 0, 'percent_change_15m': 0, 'percent_change_24h': 0, 'volume_24h': 0}

@st.cache_data(ttl=300)
def fetch_kalshi_orderbook(ticker):
    """Fetch live order book from Kalshi"""
    try:
        url = f"https://external-api.kalshi.com/trade-api/v2/markets/{ticker}/orderbook"
        response = requests.get(url, timeout=5)
        data = response.json()
        book = data.get('orderbook_fp', {})
        yes_bids = book.get('yes_dollars', [])
        no_bids = book.get('no_dollars', [])
        metrics = {
            'best_yes_bid': float(yes_bids[0][0]) if yes_bids else 0,
            'best_no_bid': float(no_bids[0][0]) if no_bids else 0,
            'yes_volume': sum([float(p[1]) for p in yes_bids[:10]]) if yes_bids else 0,
            'no_volume': sum([float(p[1]) for p in no_bids[:10]]) if no_bids else 0,
        }
        metrics['spread'] = metrics['best_yes_bid'] - metrics['best_no_bid']
        total_volume = metrics['yes_volume'] + metrics['no_volume']
        metrics['imbalance'] = (metrics['yes_volume'] - metrics['no_volume']) / (total_volume + 1)
        return metrics
    except:
        return {'best_yes_bid': 0, 'best_no_bid': 0, 'spread': 0, 'imbalance': 0}

def add_technical_indicators(df):
    """Add technical indicators"""
    df['sma_5'] = df['close'].rolling(5).mean()
    df['sma_10'] = df['close'].rolling(10).mean()
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['return_1'] = df['close'].pct_change()
    df['return_5'] = df['close'].pct_change(5)
    df['price_range'] = (df['high'] - df['low']) / df['close']
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(10).mean()
    return df

def calculate_kelly_bet(win_prob, bankroll, max_risk_pct=0.02):
    """Calculate optimal bet size"""
    odds = 1.0
    p = win_prob
    q = 1 - p
    if p <= 0.5:
        return 0
    kelly_fraction = (p * odds - q) / odds
    kelly_fraction = min(kelly_fraction, max_risk_pct)
    if kelly_fraction < 0.01:
        return 0
    return kelly_fraction * bankroll

# --- Main Loop ---
all_results = []
best_bets = []

# CoinPaprika ID mapping
paprika_map = {
    'BTC-USD': 'btc-bitcoin',
    'ETH-USD': 'eth-ethereum',
    'SOL-USD': 'sol-solana',
    'BNB-USD': 'bnb-binance-coin',
    'XRP-USD': 'xrp-xrp',
    'DOGE-USD': 'doge-dogecoin'
}

for coin in COINS:
    try:
        # 1. Fetch data
        df = fetch_yahoo_data(coin)
        if df.empty:
            continue
        
        df = add_technical_indicators(df)
        df_clean = df.dropna()
        
        if len(df_clean) < 50:
            continue
        
        # 2. Get external data
        paprika_data = fetch_coinpaprika_data(paprika_map[coin])
        coin_name = coin.replace('-USD', '')
        
        # 3. Create features
        feature_cols = ['close', 'volume', 'return_1', 'return_5', 'price_range',
                       'volume_ratio', 'rsi', 'sma_5', 'sma_10']
        
        available_cols = [col for col in feature_cols if col in df_clean.columns]
        X = df_clean[available_cols].values
        y = df_clean['close'].shift(-PREDICT_WINDOW) > df_clean['close']
        
        # Remove NaN
        X_df = pd.DataFrame(X, columns=available_cols)
        X_df['target'] = y.astype(int)
        X_df_clean = X_df.dropna()
        
        if len(X_df_clean) < 30:
            continue
        
        X_train = X_df_clean[available_cols].values
        y_train = X_df_clean['target'].values
        
        # 4. Train Random Forest (lightweight, no TensorFlow)
        model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        
        # 5. Make prediction
        last_row = df_clean[available_cols].iloc[-1].values.reshape(1, -1)
        win_prob = model.predict_proba(last_row)[0][1]
        
        # 6. Determine action
        edge = win_prob - 0.50
        bet_amount = calculate_kelly_bet(win_prob, BANKROLL, MAX_RISK_PER_TRADE)
        current_price = df_clean['close'].iloc[-1]
        
        if edge >= MIN_EDGE and win_prob > 0.55:
            action = "BUY YES" if win_prob > 0.5 else "BUY NO"
            is_signal = True
        else:
            action = "SKIP"
            is_signal = False
        
        result = {
            'Coin': coin_name,
            'Price': f"${current_price:.2f}",
            'Win_Prob': f"{win_prob:.0%}",
            'Edge': f"{edge:.0%}",
            'Change_15m': f"{paprika_data.get('percent_change_15m', 0):.1f}%",
            'Change_24h': f"{paprika_data.get('percent_change_24h', 0):.1f}%",
            'Bet_Size': f"${bet_amount:.2f}" if bet_amount > 0 else "$0.00",
            'Action': action
        }
        all_results.append(result)
        
        if is_signal:
            best_bets.append(result)
            
    except Exception as e:
        pass

# --- Display Results ---
st.subheader("📊 Predictions")

if all_results:
    df_results = pd.DataFrame(all_results)
    st.dataframe(df_results, use_container_width=True, hide_index=True)
else:
    st.warning("No results generated.")

st.divider()

# --- Best Bets ---
st.subheader("⭐ Best Bets")

if best_bets:
    for bet in best_bets:
        st.success(f"✅ **{bet['Coin']}**: {bet['Action']}")
        st.caption(f"Win Prob: {bet['Win_Prob']} | Edge: {bet['Edge']} | Bet: {bet['Bet_Size']}")
else:
    st.info("⏳ No bets meet the minimum edge threshold.")

st.divider()

# --- Data Sources ---
with st.expander("📚 Data Sources Used"):
    st.markdown("""
    ✅ **Yahoo Finance** - OHLCV price data (1-minute candles)  
    ✅ **CoinPaprika** - Price changes (15m, 24h)  
    ✅ **Kalshi** - Order book data (spread, imbalance)  
    🔧 **Random Forest** - Lightweight ML model (replaces GRU for free tier)
    """)

st.caption(f"🔮 Auto-refreshes every 60 seconds. Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
