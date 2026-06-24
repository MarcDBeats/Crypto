# ============================================
# CODE 5: CLEAN PRO KALSHI EDGE DETECTOR
# Features: Order Book (10 Levels) | Microprice | Imbalance
#           Spread Quality | Market Regime | Ternary Classification
#           No Auto-Refresh | No Backtesting
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# --- Page Config ---
st.set_page_config(
    page_title="Kalshi Edge Detector Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    .signal-summary {
        background: linear-gradient(135deg, #1e1e2f 0%, #2d2d44 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #3d3d5c;
        margin-bottom: 1rem;
    }
    .trade-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        text-decoration: none;
        font-weight: 600;
        display: inline-block;
        margin-top: 0.3rem;
    }
    .trade-button:hover {
        opacity: 0.8;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown('<div class="main-header">📊 Kalshi Edge Detector Pro</div>', unsafe_allow_html=True)
st.caption(f"⚡ Order Book (10 Levels) • Microprice • Market Regime • Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

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
        
        yes_bids = book.get('yes_dollars', [])
        no_bids = book.get('no_dollars', [])
        
        best_yes_bid = float(yes_bids[0][0]) if yes_bids else 0
        best_no_bid = float(no_bids[0][0]) if no_bids else 0
        
        spread = best_yes_bid - best_no_bid if best_yes_bid and best_no_bid else 0
        
        # Spread quality
        if spread < 0.02:
            spread_quality = "🟢 Excellent"
            spread_class = "spread-excellent"
        elif spread < 0.05:
            spread_quality = "🟡 Good"
            spread_class = "spread-good"
        else:
            spread_quality = "🔴 Poor"
            spread_class = "spread-poor"
        
        # Multi-level imbalance
        yes_vol_5 = sum([float(p[1]) for p in yes_bids[:5]]) if yes_bids else 0
        no_vol_5 = sum([float(p[1]) for p in no_bids[:5]]) if no_bids else 0
        imbalance_5 = (yes_vol_5 - no_vol_5) / (yes_vol_5 + no_vol_5 + 1) if (yes_vol_5 + no_vol_5) > 0 else 0
        
        yes_vol_10 = sum([float(p[1]) for p in yes_bids[:10]]) if yes_bids else 0
        no_vol_10 = sum([float(p[1]) for p in no_bids[:10]]) if no_bids else 0
        imbalance_10 = (yes_vol_10 - no_vol_10) / (yes_vol_10 + no_vol_10 + 1) if (yes_vol_10 + no_vol_10) > 0 else 0
        
        # Microprice
        bid_qty = yes_vol_10 if yes_vol_10 > 0 else 1
        ask_qty = no_vol_10 if no_vol_10 > 0 else 1
        microprice = (best_yes_bid * ask_qty + best_no_bid * bid_qty) / (bid_qty + ask_qty) if (bid_qty + ask_qty) > 0 else 0.50
        
        # Depth slope
        if len(yes_bids) >= 5:
            top_depth = float(yes_bids[0][1]) if yes_bids else 0
            avg_depth_5 = sum([float(p[1]) for p in yes_bids[:5]]) / 5 if len(yes_bids) >= 5 else 1
            depth_slope = top_depth / avg_depth_5 if avg_depth_5 > 0 else 1
        else:
            depth_slope = 1
        
        # Liquidity score
        spread_score = max(0, min(1, 1 - (spread / 0.10)))
        depth_score = min(1, (yes_vol_10 + no_vol_10) / 5000)
        imbalance_score = abs(imbalance_10)
        liquidity_score = (spread_score * 0.4 + depth_score * 0.3 + imbalance_score * 0.3)
        liquidity_score = max(0, min(1, liquidity_score))
        
        # Liquidity label
        if liquidity_score >= 0.7:
            liquidity_label = "🟢 High"
            liquidity_class = "liquidity-high"
        elif liquidity_score >= 0.4:
            liquidity_label = "🟡 Medium"
            liquidity_class = "liquidity-medium"
        else:
            liquidity_label = "🔴 Low"
            liquidity_class = "liquidity-low"
        
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
    
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['log_return_5'] = np.log(df['close'] / df['close'].shift(5))
    df['log_return_10'] = np.log(df['close'] / df['close'].shift(10))
    
    df['abs_log_return'] = df['log_return'].abs()
    df['abs_log_return_5'] = df['log_return_5'].abs()
    df['abs_log_return_10'] = df['log_return_10'].abs()
    df['abs_price_range'] = (df['high'] - df['low']).abs()
    df['abs_volume_change'] = df['volume'].pct_change().abs()
    
    df['sma_5'] = df['close'].rolling(5).mean()
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['ema_9'] = df['close'].ewm(span=9).mean()
    df['ema_21'] = df['close'].ewm(span=21).mean()
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['macd_abs'] = df['macd_hist'].abs()
    
    df['bb_middle'] = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * bb_std
    df['bb_lower'] = df['bb_middle'] - 2 * bb_std
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    
    df['atr'] = (df['high'] - df['low']).rolling(14).mean()
    df['volatility'] = df['log_return'].rolling(10).std()
    df['volatility_abs'] = df['volatility'].abs()
    
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(10).mean()
    df['volume_abs'] = df['volume_ratio'].abs()
    df['money_flow_index'] = ((df['close'] - df['low']) / (df['high'] - df['low'])) * df['volume']
    df['money_flow_index'] = df['money_flow_index'].rolling(14).sum() / df['volume'].rolling(14).sum() * 100
    
    df['return_1'] = df['close'].pct_change()
    df['return_5'] = df['close'].pct_change(5)
    df['return_10'] = df['close'].pct_change(10)
    df['abs_return_1'] = df['return_1'].abs()
    df['abs_return_5'] = df['return_5'].abs()
    
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
        
        prob_5m = get_model_probability(coin, 5, df_clean)
        prob_15m = get_model_probability(coin, 15, df_clean)
        
        market_price = kalshi_prices.get(coin, 0.50)
        
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
        
        # Market regime detection
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
        
        edge_5m = prob_5m - market_price if prob_5m else 0
        edge_15m = prob_15m - market_price if prob_15m else 0
        
        if liquidity_score < 0.3:
            edge_5m *= 0.6
            edge_15m *= 0.6
        elif liquidity_score < 0.5:
            edge_5m *= 0.8
            edge_15m *= 0.8
        
        # Ternary classification
        flat_threshold = FLAT_THRESHOLD
        expected_move = abs(prob_5m - 0.50) * 100 if prob_5m else 0
        
        if expected_move < flat_threshold * 100:
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

# --- SIGNAL SUMMARY BAR ---
st.markdown("---")

signals_5m = len([r for r in all_results if r['Is_Signal_5m']])
signals_15m = len([r for r in all_results if r['Is_Signal_15m']])
best_bet = best_bets[0] if best_bets else None

st.markdown("### 📊 Quick Summary")
col1, col2, col3, col4 = st.columns(4)

with col1:
    if best_bet:
        st.metric("⭐ Best Bet", f"{best_bet['Name']} — {best_bet['Direction_5m']}", delta=f"Edge: {best_bet['Edge_5m_Str']}")
    else:
        st.metric("⭐ Best Bet", "None", delta="Waiting...")

with col2:
    st.metric("📊 5m Signals", signals_5m, delta=f"{signals_5m} signals")

with col3:
    st.metric("📊 15m Signals", signals_15m, delta=f"{signals_15m} signals")

with col4:
    avg_liquidity = np.mean([r['Liquidity_Score'] for r in all_results]) if all_results else 0.5
    health_score = min(100, max(0, (avg_liquidity * 100) + 30))
    st.metric("💪 Market Health", f"{health_score:.0f}/100", delta="Good" if health_score > 60 else "Caution")

st.divider()

# --- Market Regime Banner ---
if all_results:
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
            
            # Kalshi trade link
            ticker = KALSHI_TICKERS.get(f"{result['Name']}-USD", "")
            trade_link = f"https://kalshi.com/market/{ticker}" if
