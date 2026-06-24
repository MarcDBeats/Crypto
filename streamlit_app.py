# ============================================
# CODE 4: PRO KALSHI EDGE DETECTOR
# Features: Auto-Refresh | Order Book (10 Levels) | Microprice
#           Imbalance | Spread Quality | Market Regime
#           Ternary Classification | Backtesting
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import warnings
warnings.filterwarnings('ignore')

# --- Page Config ---
st.set_page_config(
    page_title="Kalshi Edge Detector Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- AUTO-REFRESH (NEW) ---
# This makes the dashboard refresh every 30 seconds
# Using st.rerun() with a timer
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()
    
# Auto-refresh every 30 seconds
if time.time() - st.session_state.last_refresh > 30:
    st.session_state.last_refresh = time.time()
    st.rerun()

# --- Custom CSS ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e1e2f 0%, #2d2d44 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #3d3d5c;
        text-align: center;
        min-height: 200px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .metric-card .coin-name {
        font-size: 0.9rem;
        font-weight: 700;
        color: #ccc;
    }
    .metric-card .coin-price {
        font-size: 1.2rem;
        font-weight: 600;
        margin: 0.3rem 0;
    }
    .metric-card .edge-display {
        font-size: 1.3rem;
        font-weight: 700;
        margin: 0.3rem 0;
    }
    .metric-card .coin-stats {
        font-size: 0.65rem;
        color: #888;
        margin-top: 0.2rem;
    }
    .edge-positive {
        color: #00b894;
    }
    .edge-negative {
        color: #ff6b6b;
    }
    .edge-neutral {
        color: #fdcb6e;
    }
    .best-bet {
        background: linear-gradient(135deg, #00b894 0%, #00cec9 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        font-weight: 600;
    }
    .best-bet-no {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        font-weight: 600;
    }
    .skip-bet {
        background: linear-gradient(135deg, #636e72 0%, #2d3436 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: #b2bec3;
        font-weight: 600;
    }
    .regime-banner {
        text-align: center;
        padding: 0.5rem;
        border-radius: 0.5rem;
        font-weight: 700;
        font-size: 1.2rem;
        margin-bottom: 1rem;
    }
    .regime-high {
        background: #00b89433;
        border: 2px solid #00b894;
        color: #00b894;
    }
    .regime-low {
        background: #ff6b6b33;
        border: 2px solid #ff6b6b;
        color: #ff6b6b;
    }
    .regime-medium {
        background: #fdcb6e33;
        border: 2px solid #fdcb6e;
        color: #fdcb6e;
    }
    .regime-deep {
        background: #667eea33;
        border: 2px solid #667eea;
        color: #667eea;
    }
    .liquidity-high {
        color: #00b894;
        font-weight: 700;
    }
    .liquidity-medium {
        color: #fdcb6e;
        font-weight: 700;
    }
    .liquidity-low {
        color: #ff6b6b;
        font-weight: 700;
    }
    .stDataFrame {
        border-radius: 0.5rem;
        overflow: hidden;
    }
    .spread-excellent {
        color: #00b894;
        font-weight: 700;
    }
    .spread-good {
        color: #fdcb6e;
        font-weight: 700;
    }
    .spread-poor {
        color: #ff6b6b;
        font-weight: 700;
    }
    .flat-class {
        color: #fdcb6e;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown('<div class="main-header">📊 Kalshi Edge Detector Pro</div>', unsafe_allow_html=True)
st.caption(f"⚡ Auto-Refresh: 30s • Order Book (10 Levels) • Microprice • Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# --- Auto-Refresh Indicator ---
st.info(f"🔄 Auto-refreshing every 30 seconds. Next refresh at: {(datetime.now() + timedelta(seconds=30)).strftime('%H:%M:%S')}")

# --- Settings ---
BANKROLL = 100.00
MAX_RISK_PER_TRADE = 0.02
MIN_EDGE = 0.05
FLAT_THRESHOLD = 0.002  # 0.2% price change threshold for ternary classification

COINS = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD', 'DOGE-USD']

# Kalshi ticker mapping
KALSHI_TICKERS = {
    'BTC-USD': 'KXBT',
    'ETH-USD': 'KXETH',
    'SOL-USD': 'KXSOL',
    'BNB-USD': 'KXBNB',
    'XRP-USD': 'KXXRP',
    'DOGE-USD': 'KXDOGE'
}

COIN_METADATA = {
    'BTC-USD': {'name': 'Bitcoin', 'symbol': 'BTC', 'color': '#f7931a'},
    'ETH-USD': {'name': 'Ethereum', 'symbol': 'ETH', 'color': '#627eea'},
    'SOL-USD': {'name': 'Solana', 'symbol': 'SOL', 'color': '#9945ff'},
    'BNB-USD': {'name': 'BNB', 'symbol': 'BNB', 'color': '#f3ba2f'},
    'XRP-USD': {'name': 'XRP', 'symbol': 'XRP', 'color': '#00aae4'},
    'DOGE-USD': {'name': 'Dogecoin', 'symbol': 'DOGE', 'color': '#c2a633'}
}

# --- Data Fetching ---
@st.cache_data(ttl=10)
def fetch_yahoo_data(symbol):
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

@st.cache_data(ttl=10)
def fetch_kalshi_price(ticker):
    try:
        url = f"https://external-api.kalshi.com/trade-api/v2/markets/{ticker}"
        response = requests.get(url, timeout=5)
        data = response.json()
        market = data.get('market', {})
        yes_bid = market.get('yes_bid', 0)
        yes_ask = market.get('yes_ask', 0)
        if yes_bid > 0 and yes_ask > 0:
            return (yes_bid + yes_ask) / 2
        elif yes_bid > 0:
            return yes_bid
        elif yes_ask > 0:
            return yes_ask
        else:
            return None
    except:
        return None

@st.cache_data(ttl=5)
def fetch_kalshi_order_book(ticker, depth=10):
    """Fetch full order book data with depth levels"""
    try:
        url = f"https://external-api.kalshi.com/trade-api/v2/markets/{ticker}/orderbook?depth={depth}"
        response = requests.get(url, timeout=5)
        data = response.json()
        book = data.get('orderbook_fp', {})
        
        yes_bids = book.get('yes_dollars', [])  # List of [price, volume]
        no_bids = book.get('no_dollars', [])
        
        # --- 1. BEST BID/ASK ---
        best_yes_bid = float(yes_bids[0][0]) if yes_bids else 0
        best_no_bid = float(no_bids[0][0]) if no_bids else 0
        
        # --- 2. SPREAD ---
        spread = best_yes_bid - best_no_bid if best_yes_bid and best_no_bid else 0
        
        # --- 3. SPREAD QUALITY ---
        if spread < 0.02:
            spread_quality = "🟢 Excellent"
            spread_class = "spread-excellent"
        elif spread < 0.05:
            spread_quality = "🟡 Good"
            spread_class = "spread-good"
        else:
            spread_quality = "🔴 Poor"
            spread_class = "spread-poor"
        
        # --- 4. MULTI-LEVEL IMBALANCE (Top 5 levels) ---
        yes_vol_5 = sum([float(p[1]) for p in yes_bids[:5]]) if yes_bids else 0
        no_vol_5 = sum([float(p[1]) for p in no_bids[:5]]) if no_bids else 0
        imbalance_5 = (yes_vol_5 - no_vol_5) / (yes_vol_5 + no_vol_5 + 1) if (yes_vol_5 + no_vol_5) > 0 else 0
        
        # --- 5. AGGREGATE IMBALANCE (Top 10 levels) ---
        yes_vol_10 = sum([float(p[1]) for p in yes_bids[:10]]) if yes_bids else 0
        no_vol_10 = sum([float(p[1]) for p in no_bids[:10]]) if no_bids else 0
        imbalance_10 = (yes_vol_10 - no_vol_10) / (yes_vol_10 + no_vol_10 + 1) if (yes_vol_10 + no_vol_10) > 0 else 0
        
        # --- 6. MICROPRICE ---
        # Weighted average reflecting order imbalances
        bid_qty = yes_vol_10 if yes_vol_10 > 0 else 1
        ask_qty = no_vol_10 if no_vol_10 > 0 else 1
        microprice = (best_yes_bid * ask_qty + best_no_bid * bid_qty) / (bid_qty + ask_qty) if (bid_qty + ask_qty) > 0 else 0.50
        
        # --- 7. DEPTH SLOPE (Bid side) ---
        # Measures if depth is concentrated at top or spread across levels
        if len(yes_bids) >= 5:
            top_depth = float(yes_bids[0][1]) if yes_bids else 0
            avg_depth_5 = sum([float(p[1]) for p in yes_bids[:5]]) / 5 if len(yes_bids) >= 5 else 1
            depth_slope = top_depth / avg_depth_5 if avg_depth_5 > 0 else 1
        else:
            depth_slope = 1
        
        # --- 8. LIQUIDITY SCORE (Enhanced) ---
        # Combines spread, depth, and imbalance
        spread_score = max(0, min(1, 1 - (spread / 0.10)))
        depth_score = min(1, (yes_vol_10 + no_vol_10) / 5000)  # 5000 = high liquidity benchmark
        imbalance_score = abs(imbalance_10)  # 0-1
        liquidity_score = (spread_score * 0.4 + depth_score * 0.3 + imbalance_score * 0.3)
        liquidity_score = max(0, min(1, liquidity_score))
        
        # --- 9. LIQUIDITY LABEL ---
        if liquidity_score >= 0.7:
            liquidity_label = "🟢 High"
            liquidity_class = "liquidity-high"
        elif liquidity_score >= 0.4:
            liquidity_label = "🟡 Medium"
            liquidity_class = "liquidity-medium"
        else:
            liquidity_label = "🔴 Low"
            liquidity_class = "liquidity-low"
        
        # --- 10. DEPTH (Total levels) ---
        depth = len(yes_bids) + len(no_bids)
        
        return {
            'best_yes_bid': best_yes_bid,
            'best_no_bid': best_no_bid,
            'spread': spread,
            'spread_quality': spread_quality,
            'spread_class': spread_class,
            'imbalance_5': imbalance_5,
            'imbalance_10': imbalance_10,
            'microprice': microprice,
            'depth_slope': depth_slope,
            'liquidity_score': liquidity_score,
            'liquidity_label': liquidity_label,
            'liquidity_class': liquidity_class,
            'depth': depth,
            'yes_volume': yes_vol_10,
            'no_volume': no_vol_10,
            'top_bids': yes_bids[:10] if yes_bids else [],
            'top_asks': no_bids[:10] if no_bids else []
        }
    except:
        return {
            'best_yes_bid': 0,
            'best_no_bid': 0,
            'spread': 0,
            'spread_quality': '🔴 Poor',
            'spread_class': 'spread-poor',
            'imbalance_5': 0,
            'imbalance_10': 0,
            'microprice': 0.50,
            'depth_slope': 1,
            'liquidity_score': 0,
            'liquidity_label': '🔴 Low',
            'liquidity_class': 'liquidity-low',
            'depth': 0,
            'yes_volume': 0,
            'no_volume': 0,
            'top_bids': [],
            'top_asks': []
        }

@st.cache_data(ttl=3600)
def fetch_fear_greed_index():
    try:
        url = "https://api.alternative.me/fng/?limit=2"
        response = requests.get(url, timeout=5)
        data = response.json()
        if data and 'data' in data:
            return {
                'value': int(data['data'][0]['value']),
                'classification': data['data'][0]['value_classification']
            }
        return {'value': 50, 'classification': 'Neutral'}
    except:
        return {'value': 50, 'classification': 'Neutral'}

# --- Enhanced Technical Indicators ---
def add_enhanced_indicators(df):
    """Add comprehensive technical indicators with Log Returns and Absolute Features"""
    
    # --- 1. LOG RETURNS ---
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['log_return_5'] = np.log(df['close'] / df['close'].shift(5))
    df['log_return_10'] = np.log(df['close'] / df['close'].shift(10))
    
    # --- 2. ABSOLUTE FEATURES ---
    df['abs_log_return'] = df['log_return'].abs()
    df['abs_log_return_5'] = df['log_return_5'].abs()
    df['abs_log_return_10'] = df['log_return_10'].abs()
    df['abs_price_range'] = (df['high'] - df['low']).abs()
    df['abs_volume_change'] = df['volume'].pct_change().abs()
    
    # --- 3. MOVING AVERAGES ---
    df['sma_5'] = df['close'].rolling(5).mean()
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['ema_9'] = df['close'].ewm(span=9).mean()
    df['ema_21'] = df['close'].ewm(span=21).mean()
    
    # --- 4. RSI ---
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # --- 5. MACD ---
    df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['macd_abs'] = df['macd_hist'].abs()
    
    # --- 6. BOLLINGER BANDS ---
    df['bb_middle'] = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * bb_std
    df['bb_lower'] = df['bb_middle'] - 2 * bb_std
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    
    # --- 7. VOLATILITY ---
    df['atr'] = (df['high'] - df['low']).rolling(14).mean()
    df['volatility'] = df['log_return'].rolling(10).std()
    df['volatility_abs'] = df['volatility'].abs()
    
    # --- 8. VOLUME FEATURES ---
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(10).mean()
    df['volume_abs'] = df['volume_ratio'].abs()
    df['money_flow_index'] = ((df['close'] - df['low']) / (df['high'] - df['low'])) * df['volume']
    df['money_flow_index'] = df['money_flow_index'].rolling(14).sum() / df['volume'].rolling(14).sum() * 100
    
    # --- 9. RETURN FEATURES ---
    df['return_1'] = df['close'].pct_change()
    df['return_5'] = df['close'].pct_change(5)
    df['return_10'] = df['close'].pct_change(10)
    df['abs_return_1'] = df['return_1'].abs()
    df['abs_return_5'] = df['return_5'].abs()
    
    # --- 10. PRICE RATIO ---
    df['price_range'] = (df['high'] - df['low']) / df['close']
    df['abs_price_range'] = df['price_range'].abs()
    df['high_low_ratio'] = df['high'] / df['low']
    df['close_open_ratio'] = df['close'] / df['open']
    
    return df

# --- GET MODEL PROBABILITY ---
def get_model_probability(coin_symbol, window_minutes, df_clean=None):
    try:
        if df_clean is None:
            df = fetch_yahoo_data(coin_symbol)
            if df.empty:
                return None
            df = add_enhanced_indicators(df)
            df_clean = df.dropna()
        
        if len(df_clean) < 60:
            return None
        
        feature_cols = [
            'close', 'volume', 
            'log_return', 'abs_log_return',
            'sma_5', 'sma_10', 
            'rsi', 'macd_hist', 
            'bb_position', 'atr', 
            'return_1', 'return_5'
        ]
        
        available_cols = [col for col in feature_cols if col in df_clean.columns]
        
        if len(available_cols) < 5:
            return None
        
        X = df_clean[available_cols].values
        y = df_clean['close'].shift(-window_minutes) > df_clean['close']
        
        X_df = pd.DataFrame(X, columns=available_cols)
        X_df['target'] = y.astype(int)
        X_df_clean = X_df.dropna()
        
        if len(X_df_clean) < 20:
            return None
        
        X_train = X_df_clean[available_cols].values
        y_train = X_df_clean['target'].values
        
        try:
            from xgboost import XGBClassifier
            model = XGBClassifier(n_estimators=30, max_depth=3, learning_rate=0.1, random_state=42, use_label_encoder=False)
            model.fit(X_train, y_train)
        except:
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=30, max_depth=5, random_state=42)
            model.fit(X_train, y_train)
        
        last_row = df_clean[available_cols].iloc[-1].values.reshape(1, -1)
        win_prob = model.predict_proba(last_row)[0][1]
        
        return win_prob
    except Exception as e:
        return None

# --- MAIN LOOP ---
all_results = []
best_bets = []

fear_greed = fetch_fear_greed_index()

# Fetch Kalshi prices and order books
kalshi_prices = {}
kalshi_order_books = {}
for coin in COINS:
    ticker = KALSHI_TICKERS.get(coin)
    if ticker:
        price = fetch_kalshi_price(ticker)
        if price:
            kalshi_prices[coin] = price
        order_book = fetch_kalshi_order_book(ticker, depth=10)
        if order_book:
            kalshi_order_books[coin] = order_book

progress_bar = st.progress(0)
status_text = st.empty()

for idx, coin in enumerate(COINS):
    status_text.text(f"🔄 Analyzing {coin} with order book data...")
    
    try:
        df = fetch_yahoo_data(coin)
        if df.empty:
            continue
        
        df = add_enhanced_indicators(df)
        df_clean = df.dropna()
        
        if len(df_clean) < 60:
            continue
        
        # Get model probabilities
        prob_5m = get_model_probability(coin, 5, df_clean)
        prob_15m = get_model_probability(coin, 15, df_clean)
        
        # Get Kalshi market price (with fallback)
        market_price = kalshi_prices.get(coin, 0.50)
        
        # Get order book data
        order_book = kalshi_order_books.get(coin, {})
        spread = order_book.get('spread', 0)
        spread_quality = order_book.get('spread_quality', '🔴 Poor')
        liquidity_score = order_book.get('liquidity_score', 0)
        liquidity_label = order_book.get('liquidity_label', '🔴 Low')
        imbalance_5 = order_book.get('imbalance_5', 0)
        imbalance_10 = order_book.get('imbalance_10', 0)
        microprice = order_book.get('microprice', 0.50)
        depth_slope = order_book.get('depth_slope', 1)
        depth = order_book.get('depth', 0)
        yes_volume = order_book.get('yes_volume', 0)
        no_volume = order_book.get('no_volume', 0)
        
        # --- MARKET REGIME DETECTION ---
        # Based on activity and liquidity
        if liquidity_score >= 0.7 and imbalance_10 > 0.3:
            regime = "🟢 High Activity / Deep Liquidity"
            regime_class = "regime-high"
        elif liquidity_score >= 0.5 and imbalance_10 > 0.15:
            regime = "🟡 Moderate Activity / Good Liquidity"
            regime_class = "regime-medium"
        elif liquidity_score >= 0.3:
            regime = "🟠 Low Activity / Adequate Liquidity"
            regime_class = "regime-low"
        else:
            regime = "🔴 Very Low Activity / Poor Liquidity"
            regime_class = "regime-low"
        
        # Calculate edges
        edge_5m = prob_5m - market_price if prob_5m else 0
        edge_15m = prob_15m - market_price if prob_15m else 0
        
        # Adjust edge based on liquidity
        if liquidity_score < 0.3:
            edge_5m *= 0.6
            edge_15m *= 0.6
        elif liquidity_score < 0.5:
            edge_5m *= 0.8
            edge_15m *= 0.8
        
        # --- TERNARY CLASSIFICATION ---
        # Uses FLAT_THRESHOLD to determine if price is likely to move
        # This reduces false positives on small fluctuations
        flat_threshold = FLAT_THRESHOLD
        
        # Calculate expected move based on model probability and market price
        # This is a simplified version — you can enhance this further
        expected_move = abs(prob_5m - 0.50) * 100  # Percentage expected move
        
        # --- DECISION LOGIC ---
        if expected_move < flat_threshold * 100:  # Convert threshold to percentage
            decision_5m = "FLAT"
            direction_5m = "FLAT"
            is_signal_5m = False
        elif prob_5m and edge_5m >= MIN_EDGE:
            decision_5m = "BUY YES"
            direction_5m = "YES"
            is_signal_5m = True
        elif prob_5m and edge_5m <= -MIN_EDGE:
            decision_5m = "BUY NO"
            direction_5m = "NO"
            is_signal_5m = True
        else:
            decision_5m = "SKIP"
            direction_5m = "WAIT"
            is_signal_5m = False
        
        if expected_move < flat_threshold * 100:
            decision_15m = "FLAT"
            direction_15m = "FLAT"
            is_signal_15m = False
        elif prob_15m and edge_15m >= MIN_EDGE:
            decision_15m = "BUY YES"
            direction_15m = "YES"
            is_signal_15m = True
        elif prob_15m and edge_15m <= -MIN_EDGE:
            decision_15m = "BUY NO"
            direction_15m = "NO"
            is_signal_15m = True
        else:
            decision_15m = "SKIP"
            direction_15m = "WAIT"
            is_signal_15m = False
        
        # Calculate actual changes
        if len(df_clean) > 5:
            change_5m = ((df_clean['close'].iloc[-1] - df_clean['close'].iloc[-6]) / df_clean['close'].iloc[-6]) * 100
        else:
            change_5m = 0
        
        if len(df_clean) > 15:
            change_15m = ((df_clean['close'].iloc[-1] - df_clean['close'].iloc[-16]) / df_clean['close'].iloc[-16]) * 100
        else:
            change_15m = 0
        
        metadata = COIN_METADATA.get(coin, {'name': coin.replace('-USD', ''), 'symbol': coin.replace('-USD', ''), 'color': '#ffffff'})
        current_price = df_clean['close'].iloc[-1]
        
        result = {
            'Name': metadata['name'],
            'Symbol': metadata['symbol'],
            'Price': current_price,
            'Price_Str': f"${current_price:.2f}",
            'Market_Price': market_price,
            'Market_Price_Str': f"{market_price:.3f}",
            'Spread': spread,
            'Spread_Str': f"{spread:.3f}",
            'Spread_Quality': spread_quality,
            'Imbalance_5': imbalance_5,
            'Imbalance_5_Str': f"{imbalance_5:.2f}",
            'Imbalance_10': imbalance_10,
            'Imbalance_10_Str': f"{imbalance_10:.2f}",
            'Microprice': microprice,
            'Microprice_Str': f"{microprice:.3f}",
            'Depth_Slope': depth_slope,
            'Depth_Slope_Str': f"{depth_slope:.2f}",
            'Liquidity': liquidity_label,
            'Liquidity_Score': liquidity_score,
            'Depth': depth,
            'Yes_Volume': yes_volume,
            'No_Volume': no_volume,
            'Regime': regime,
            'Regime_Class': regime_class,
            'Prob_5m': prob_5m,
            'Prob_5m_Str': f"{prob_5m:.0%}" if prob_5m else '—',
            'Edge_5m': edge_5m,
            'Edge_5m_Str': f"{edge_5m:.0%}" if prob_5m else '—',
            'Decision_5m': decision_5m,
            'Direction_5m': direction_5m,
            'Is_Signal_5m': is_signal_5m,
            'Prob_15m': prob_15m,
            'Prob_15m_Str': f"{prob_15m:.0%}" if prob_15m else '—',
            'Edge_15m': edge_15m,
            'Edge_15m_Str': f"{edge_15m:.0%}" if prob_15m else '—',
            'Decision_15m': decision_15m,
            'Direction_15m': direction_15m,
            'Is_Signal_15m': is_signal_15m,
            'Change_5m': change_5m,
            'Change_15m': change_15m,
            'Color': metadata['color']
        }
        all_results.append(result)
        
        if is_signal_5m or is_signal_15m:
            best_bets.append(result)
            
    except Exception as e:
        pass
    
    progress_bar.progress((idx + 1) / len(COINS))

status_text.empty()
progress_bar.empty()

# ============================================
# 📊 DASHBOARD DISPLAY
# ============================================

# --- Market Regime Banner ---
if all_results:
    # Aggregate regime from first coin (or average)
    first_coin = all_results[0] if all_results else {}
    regime_display = first_coin.get('Regime', '🟡 Moderate Activity / Good Liquidity')
    regime_class = first_coin.get('Regime_Class', 'regime-medium')
    
    st.markdown(f"""
    <div class="regime-banner {regime_class}">
        📊 Market Regime: {regime_display}
    </div>
    """, unsafe_allow_html=True)

# --- Fear & Greed ---
st.markdown("### 🧠 Market Sentiment")
fg_col1, fg_col2, fg_col3, fg_col4, fg_col5 = st.columns([1, 1, 2, 1, 1])
with fg_col3:
    fg_color = '#00b894' if fear_greed['value'] >= 60 else '#fdcb6e' if fear_greed['value'] >= 40 else '#ff6b6b'
    st.markdown(f"""
    <div style="text-align: center; padding: 0.5rem; border-radius: 0.5rem; border: 2px solid {fg_color};">
        <span style="font-size: 1.5rem;">{fear_greed['classification']}</span><br>
        <span style="font-size: 2.5rem; font-weight: 700; color: {fg_color};">{fear_greed['value']}</span><br>
        <span style="font-size: 0.8rem; color: #888;">Fear & Greed Index</span>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# --- Metric Cards ---
st.markdown("### 📊 Market Overview")
m_cols = st.columns(6)

for i, result in enumerate(all_results):
    if i < 6:
        with m_cols[i]:
            edge = result['Edge_5m']
            if edge >= MIN_EDGE:
                edge_class = "edge-positive"
                edge_display = f"🟢 +{edge:.0%}"
            elif edge <= -MIN_EDGE:
                edge_class = "edge-negative"
                edge_display = f"🔴 {edge:.0%}"
            else:
                edge_class = "edge-neutral"
                edge_display = f"🟡 {edge:.0%}"
            
            # Direction display with FLAT handling
            if result['Direction_5m'] == "FLAT":
                dir_display = "⏸️ FLAT"
                dir_color = "#fdcb6e"
            elif result['Direction_5m'] == "YES":
                dir_display = "⬆️ YES"
                dir_color = "#00b894"
            elif result['Direction_5m'] == "NO":
                dir_display = "⬇️ NO"
                dir_color = "#ff6b6b"
            else:
                dir_display = "⏳ WAIT"
                dir_color = "#888"
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="coin-name">{result['Name']} ({result['Symbol']})</div>
                <div class="coin-price">{result['Price_Str']}</div>
                <div style="font-size: 0.8rem; color: #888;">
                    Kalshi: {result['Market_Price_Str']}
                </div>
                <div class="edge-display {edge_class}">
                    Edge: {edge_display}
                </div>
                <div style="display: flex; justify-content: center; gap: 0.5rem; margin-top: 0.3rem;">
                    <span style="font-size: 0.7rem; color: #888;">5m:</span>
                    <span style="font-size: 0.8rem; font-weight: 600; color: {dir_color};">
                        {dir_display}
                    </span>
                    <span style="font-size: 0.7rem; color: #888;">| 15m:</span>
                    <span style="font-size: 0.8rem; font-weight: 600; color: {'#00b894' if result['Direction_15m'] == 'YES' else '#ff6b6b' if result['Direction_15m'] == 'NO' else '#fdcb6e' if result['Direction_15m'] == 'FLAT' else '#888'};">
                        {'⬆️ YES' if result['Direction_15m'] == 'YES' else '⬇️ NO' if result['Direction_15m'] == 'NO' else '⏸️ FLAT' if result['Direction_15m'] == 'FLAT' else '⏳ WAIT'}
                    </span>
                </div>
                <div class="coin-stats">
                    Model 5m: {result['Prob_5m_Str']} | 15m: {result['Prob_15m_Str']}
                </div>
                <div class="coin-stats">
                    Liquidity: {result['Liquidity']} | Spread: {result['Spread_Quality']}
                </div>
            </div>
            """, unsafe_allow_html=True)

st.divider()

# --- Detailed Table ---
st.markdown("### 📈 Detailed Edge Analysis (with Order Book Data)")

if all_results:
    df_display = pd.DataFrame([{
        'Coin': r['Name'],
        'Symbol': r['Symbol'],
        'Price': r['Price_Str'],
        'Kalshi Price': r['Market_Price_Str'],
        'Spread': r['Spread_Str'],
        'Spread Quality': r['Spread_Quality'],
        'Imbalance (5)': r['Imbalance_5_Str'],
        'Imbalance (10)': r['Imbalance_10_Str'],
        'Microprice': r['Microprice_Str'],
        'Liquidity': r['Liquidity'],
        'Regime': r['Regime'],
        'Model 5m': r['Prob_5m_Str'],
        'Edge 5m': r['Edge_5m_Str'],
        'Decision 5m': r['Decision_5m'],
        'Model 15m': r['Prob_15m_Str'],
        'Edge 15m': r['Edge_15m_Str'],
        'Decision 15m': r['Decision_15m'],
        '5m Change': f"{r['Change_5m']:+.1f}%",
        '15m Change': f"{r['Change_15m']:+.1f}%"
    } for r in all_results])
    
    st.dataframe(df_display, use_container_width=True, hide_index=True)
else:
    st.warning("No predictions available. Please check data sources.")

st.divider()

# --- Best Bets ---
st.markdown("### ⭐ Best Bets")

if best_bets:
    cols = st.columns(min(len(best_bets), 3))
    for i, bet in enumerate(best_bets[:6]):
        col = cols[i % 3]
        
        if bet['Is_Signal_5m']:
            decision = bet['Decision_5m']
            direction = bet['Direction_5m']
            prob = bet['Prob_5m_Str']
            edge = bet['Edge_5m_Str']
            timeframe = "5m"
        elif bet['Is_Signal_15m']:
            decision = bet['Decision_15m']
            direction = bet['Direction_15m']
            prob = bet['Prob_15m_Str']
            edge = bet['Edge_15m_Str']
            timeframe = "15m"
        else:
            continue
        
        if "YES" in decision:
            card_class = "best-bet"
            emoji = "🟢"
        elif "NO" in decision:
            card_class = "best-bet-no"
            emoji = "🔴"
        else:
            continue
        
        # Liquidity indicator
        if bet['Liquidity_Score'] >= 0.7:
            liquidity_emoji = "🟢"
        elif bet['Liquidity_Score'] >= 0.4:
            liquidity_emoji = "🟡"
        else:
            liquidity_emoji = "🔴"
        
        with col:
            st.markdown(f"""
            <div class="{card_class}">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 1.2rem;">{bet['Name']} ({bet['Symbol']})</span>
                    <span style="font-size: 1.5rem;">{emoji} {direction}</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-top: 0.5rem;">
                    <span>Decision: {decision}</span>
                    <span>Edge: {edge}</span>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 0.8rem; opacity: 0.8;">
                    <span>Time: {timeframe}</span>
                    <span>Model: {prob}</span>
                    <span>Kalshi: {bet['Market_Price_Str']}</span>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 0.8rem; opacity: 0.8;">
                    <span>Spread: {bet['Spread_Str']}</span>
                    <span>Liquidity: {liquidity_emoji} {bet['Liquidity']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="skip-bet">
        ⏳ No edges meet the minimum threshold. Waiting for better opportunities...
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ============================================
# 🧪 BACKTESTING ENGINE
# ============================================

st.markdown("### 🧪 Backtest Your Strategy")
st.caption("Test your model on historical data to see how it would have performed.")

# --- Backtest Settings ---
backtest_col1, backtest_col2, backtest_col3, backtest_col4 = st.columns(4)

with backtest_col1:
    backtest_coin = st.selectbox("Coin:", COINS, key="backtest_coin")
with backtest_col2:
    backtest_period = st.selectbox("Period:", ["7 Days", "14 Days", "30 Days", "60 Days", "90 Days"], key="backtest_period")
with backtest_col3:
    backtest_min_edge = st.number_input("Min Edge (%):", min_value=0, max_value=20, value=5, key="backtest_edge") / 100
with backtest_col4:
    backtest_bet_type = st.selectbox("Bet Size:", ["Fixed $1", "Fixed $5", "Kelly Criterion"], key="backtest_bet")

# --- Run Backtest Button ---
if st.button("▶️ Run Backtest", type="primary"):
    with st.spinner("Running backtest... This may take 30-60 seconds."):
        
        # --- 1. Fetch historical data ---
        days_map = {"7 Days": 7, "14 Days": 14, "30 Days": 30, "60 Days": 60, "90 Days": 90}
        days = days_map[backtest_period]
        
        ticker = yf.Ticker(backtest_coin)
        df_hist = ticker.history(period=f"{days + 2}d", interval="1m")
        
        if df_hist.empty:
            st.error("No historical data available. Please try another coin.")
        else:
            df_hist = df_hist.reset_index()
            df_hist = df_hist.rename(columns={
                'Datetime': 'time',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            df_hist = add_enhanced_indicators(df_hist)
            df_hist_clean = df_hist.dropna()
            
            if len(df_hist_clean) < 100:
                st.error("Not enough clean data for backtesting. Try a longer period.")
            else:
                # --- 2. Run backtest simulation ---
                trades = []
                equity_curve = []
                balance = 100.00
                window_size = 60
                
                for i in range(window_size, len(df_hist_clean) - 15):
                    data_slice = df_hist_clean.iloc[:i+1]
                    
                    if len(data_slice) < 60:
                        continue
                    
                    feature_cols = [
                        'close', 'volume', 
                        'log_return', 'abs_log_return',
                        'sma_5', 'sma_10', 
                        'rsi', 'macd_hist', 
                        'bb_position', 'atr', 
                        'return_1', 'return_5'
                    ]
                    
                    available_cols = [col for col in feature_cols if col in data_slice.columns]
                    
                    if len(available_cols) < 5:
                        continue
                    
                    X = data_slice[available_cols].values
                    y = data_slice['close'].shift(-5) > data_slice['close']
                    
                    X_df = pd.DataFrame(X, columns=available_cols)
                    X_df['target'] = y.astype(int)
                    X_df_clean = X_df.dropna()
                    
                    if len(X_df_clean) < 20:
                        continue
                    
                    X_train = X_df_clean[available_cols].values
                    y_train = X_df_clean['target'].values
                    
                    try:
                        from xgboost import XGBClassifier
                        model = XGBClassifier(n_estimators=30, max_depth=3, learning_rate=0.1, random_state=42, use_label_encoder=False)
                        model.fit(X_train, y_train)
                    except:
                        from sklearn.ensemble import RandomForestClassifier
                        model = RandomForestClassifier(n_estimators=30, max_depth=5, random_state=42)
                        model.fit(X_train, y_train)
                    
                    last_row = data_slice[available_cols].iloc[-1].values.reshape(1, -1)
                    win_prob = model.predict_proba(last_row)[0][1]
                    
                    current_price = data_slice['close'].iloc[-1]
                    market_price = 0.50
                    
                    edge = win_prob - market_price
                    
                    if edge >= backtest_min_edge:
                        action = "BUY YES"
                        direction = "YES"
                    elif edge <= -backtest_min_edge:
                        action = "BUY NO"
                        direction = "NO"
                    else:
                        action = "SKIP"
                        direction = "WAIT"
                    
                    if action != "SKIP":
                        if backtest_bet_type == "Kelly Criterion":
                            bet_size = (abs(edge) / (1 - market_price)) * balance
                            bet_size = min(bet_size, balance * 0.02)
                        elif backtest_bet_type == "Fixed $1":
                            bet_size = 1.00
                        else:
                            bet_size = 5.00
                        
                        bet_size = max(bet_size, 0.01)
                        bet_size = min(bet_size, balance)
                        
                        entry_price = current_price
                        entry_time = data_slice['time'].iloc[-1]
                        
                        future_price = data_slice['close'].iloc[i+5] if i+5 < len(data_slice) else current_price
                        
                        if direction == "YES":
                            win = future_price > entry_price
                        else:
                            win = future_price < entry_price
                        
                        if win:
                            profit = bet_size * 0.95
                        else:
                            profit = -bet_size
                        
                        balance += profit
                        
                        trades.append({
                            'time': entry_time,
                            'price': entry_price,
                            'future_price': future_price,
                            'direction': direction,
                            'win_prob': win_prob,
                            'edge': edge,
                            'bet_size': bet_size,
                            'profit': profit,
                            'win': win
                        })
                    
                    equity_curve.append({
                        'time': data_slice['time'].iloc[-1],
                        'balance': balance
                    })
                
                # --- 5. Calculate results ---
                if trades:
                    df_trades = pd.DataFrame(trades)
                    df_equity = pd.DataFrame(equity_curve)
                    
                    wins = df_trades[df_trades['win'] == True]
                    losses = df_trades[df_trades['win'] == False]
                    
                    total_trades = len(df_trades)
                    win_rate = len(wins) / total_trades if total_trades > 0 else 0
                    total_profit = df_trades['profit'].sum()
                    max_drawdown = (df_equity['balance'].cummax() - df_equity['balance']).max()
                    final_balance = df_equity['balance'].iloc[-1] if not df_equity.empty else 100
                    total_return = ((final_balance - 100) / 100) * 100
                    
                    returns = df_equity['balance'].pct_change().dropna()
                    sharpe = (returns.mean() / returns.std() * np.sqrt(252 * 6.5 * 12)) if returns.std() > 0 else 0
                    
                    st.success("Backtest Complete!")
                    
                    res_col1, res_col2, res_col3, res_col4 = st.columns(4)
                    res_col1.metric("Total Trades", total_trades)
                    res_col2.metric("Win Rate", f"{win_rate:.0%}")
                    res_col3.metric("Total Profit", f"${total_profit:.2f}", delta=f"{total_return:.1f}%")
                    res_col4.metric("Final Balance", f"${final_balance:.2f}")
                    
                    res2_col1, res2_col2, res2_col3, res2_col4 = st.columns(4)
                    res2_col1.metric("Avg Profit/Trade", f"${df_trades['profit'].mean():.2f}")
                    res2_col2.metric("Max Drawdown", f"${max_drawdown:.2f}")
                    res2_col3.metric("Sharpe Ratio", f"{sharpe:.2f}")
                    res2_col4.metric("Total Return", f"{total_return:+.1f}%")
                    
                    st.subheader("📈 Equity Curve")
                    fig_equity = go.Figure()
                    fig_equity.add_trace(go.Scatter(
                        x=df_equity['time'],
                        y=df_equity['balance'],
                        mode='lines',
                        name='Balance',
                        line=dict(color='#00b894', width=2)
                    ))
                    fig_equity.add_hline(y=100, line_dash="dash", line_color="gray")
                    fig_equity.update_layout(
                        title="Portfolio Balance Over Time",
                        xaxis_title="Time",
                        yaxis_title="Balance ($)",
                        height=300,
                        template='plotly_dark'
                    )
                    st.plotly_chart(fig_equity, use_container_width=True)
                    
                    st.subheader("📊 Trade Distribution")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        fig_wins = go.Figure(data=[go.Pie(
                            labels=['Wins', 'Losses'],
                            values=[len(wins), len(losses)],
                            marker_colors=['#00b894', '#ff6b6b'],
                            hole=0.4
                        )])
                        fig_wins.update_layout(height=300, template='plotly_dark')
                        st.plotly_chart(fig_wins, use_container_width=True)
                    
                    with col2:
                        fig_profit = go.Figure(data=[go.Histogram(
                            x=df_trades['profit'],
                            nbinsx=20,
                            marker_color='#667eea'
                        )])
                        fig_profit.update_layout(
                            title="Profit per Trade",
                            xaxis_title="Profit ($)",
                            yaxis_title="Count",
                            height=300,
                            template='plotly_dark'
                        )
                        st.plotly_chart(fig_profit, use_container_width=True)
                    
                    with st.expander("📋 View All Trades"):
                        df_display_trades = df_trades.copy()
                        df_display_trades['time'] = df_display_trades['time'].dt.strftime('%Y-%m-%d %H:%M')
                        df_display_trades['win'] = df_display_trades['win'].map({True: '✅ WIN', False: '❌ LOSS'})
                        df_display_trades = df_display_trades.rename(columns={
                            'time': 'Time',
                            'direction': 'Direction',
                            'win_prob': 'Win Prob',
                            'edge': 'Edge',
                            'bet_size': 'Bet Size',
                            'profit': 'Profit',
                            'win': 'Result'
                        })
                        df_display_trades['Win Prob'] = df_display_trades['Win Prob'].apply(lambda x: f"{x:.0%}")
                        df_display_trades['Edge'] = df_display_trades['Edge'].apply(lambda x: f"{x:.0%}")
                        df_display_trades['Bet Size'] = df_display_trades['Bet Size'].apply(lambda x: f"${x:.2f}")
                        df_display_trades['Profit'] = df_display_trades['Profit'].apply(lambda x: f"${x:.2f}")
                        st.dataframe(df_display_trades, use_container_width=True, hide_index=True)
                    
                else:
                    st.warning("No trades were generated during the backtest period. Try adjusting the minimum edge or using a different coin.")

st.divider()

# --- Sidebar ---
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown(f"**Bankroll:** ${BANKROLL:.2f}")
    st.markdown(f"**Max Risk/Trade:** {MAX_RISK_PER_TRADE*100:.0f}%")
    st.markdown(f"**Min Edge:** {MIN_EDGE*100:.0f}%")
    st.markdown(f"**Flat Threshold:** {FLAT_THRESHOLD*100:.1f}%")
    
    st.divider()
    
    st.markdown("### 🆕 Code 4 Features")
    st.markdown("""
    ✅ **Auto-Refresh** (30 seconds)  
    ✅ **Order Book (10 Levels)**  
    ✅ **Aggregate Imbalance** (Top 5 & 10 levels)  
    ✅ **Microprice** (Weighted average)  
    ✅ **Spread Quality** (Excellent/Good/Poor)  
    ✅ **Enhanced Liquidity Score**  
    ✅ **Market Regime Banner**  
    ✅ **Ternary Classification** (UP/DOWN/FLAT)  
    ✅ **Depth Slope Analysis**  
    ✅ **Backtesting Engine**  
    """)
    
    st.divider()
    
    st.markdown("### 🎯 How It Works")
    st.markdown("""
    1. **Model Estimate** — Enhanced ML predicts probability  
    2. **Kalshi Price** — Market's implied probability  
    3. **Order Book** — 10 levels of bid/ask data  
    4. **Microstructure Features** — Imbalance, Microprice, Depth  
    5. **Ternary Classification** — UP if edge > threshold, DOWN if edge < -threshold, FLAT otherwise  
    6. **Decision**:
       - Edge > 5% → **BUY YES**
       - Edge < -5% → **BUY NO**
       - Otherwise → **SKIP** or **FLAT**
    """)
    
    st.divider()
    
    st.markdown("### 📚 Data Sources")
    st.markdown("""
    ✅ Yahoo Finance (Price Data)  
    ✅ Kalshi API (Price + Order Book)  
    ✅ Alternative.me (Fear & Greed)  
    ✅ XGBoost/Random Forest (ML)  
    """)
    
    st.divider()
    
    st.markdown("### 🔄 Auto-Refresh")
    st.caption("Refreshes every 30 seconds")

# --- Footer ---
st.divider()
st.caption(f"⚡ Code 4 • Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} • Auto-Refresh: 30s • Order Book (10 Levels) • Microprice")
