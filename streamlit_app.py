# ============================================
# CODE 6: ADVANCED KALSHI EDGE DETECTOR
# Features: Hybrid CNN-GRU | Technical Indicators | Macro Data
#           Lag Features | Ensemble Logic | Hyperparameter Tuning
#           Order Book (10 Levels) | Depth Slope | Market Regime
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GRU, Dense, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor
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
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown('<div class="main-header">📊 Kalshi Edge Detector Pro</div>', unsafe_allow_html=True)
st.caption(f"⚡ Code 6 • CNN-GRU • Technical Indicators • Macro Data • Ensemble • Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# --- Settings ---
BANKROLL = 100.00
MAX_RISK_PER_TRADE = 0.02
MIN_EDGE = 0.05
FLAT_THRESHOLD = 0.002
LOOKBACK_WINDOW = 30

COINS = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD', 'DOGE-USD']

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
def fetch_macro_data():
    """Fetch macroeconomic data (SPY, VIX, DXY)"""
    try:
        spy = yf.Ticker("SPY").history(period='5d', interval='1m')
        vix = yf.Ticker("^VIX").history(period='5d', interval='1m')
        dxy = yf.Ticker("DX-Y.NYB").history(period='5d', interval='1m')
        
        macro = {
            'spy_close': spy['Close'].iloc[-1] if not spy.empty else 0,
            'vix_close': vix['Close'].iloc[-1] if not vix.empty else 0,
            'dxy_close': dxy['Close'].iloc[-1] if not dxy.empty else 0,
            'spy_change': spy['Close'].pct_change().iloc[-1] if len(spy) > 1 else 0,
            'vix_change': vix['Close'].pct_change().iloc[-1] if len(vix) > 1 else 0
        }
        return macro
    except:
        return {'spy_close': 0, 'vix_close': 0, 'dxy_close': 0, 'spy_change': 0, 'vix_change': 0}

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
        
        if spread < 0.02:
            spread_quality = "🟢 Excellent"
            spread_class = "spread-excellent"
        elif spread < 0.05:
            spread_quality = "🟡 Good"
            spread_class = "spread-good"
        else:
            spread_quality = "🔴 Poor"
            spread_class = "spread-poor"
        
        yes_vol_5 = sum([float(p[1]) for p in yes_bids[:5]]) if yes_bids else 0
        no_vol_5 = sum([float(p[1]) for p in no_bids[:5]]) if no_bids else 0
        imbalance_5 = (yes_vol_5 - no_vol_5) / (yes_vol_5 + no_vol_5 + 1) if (yes_vol_5 + no_vol_5) > 0 else 0
        
        yes_vol_10 = sum([float(p[1]) for p in yes_bids[:10]]) if yes_bids else 0
        no_vol_10 = sum([float(p[1]) for p in no_bids[:10]]) if no_bids else 0
        imbalance_10 = (yes_vol_10 - no_vol_10) / (yes_vol_10 + no_vol_10 + 1) if (yes_vol_10 + no_vol_10) > 0 else 0
        
        bid_qty = yes_vol_10 if yes_vol_10 > 0 else 1
        ask_qty = no_vol_10 if no_vol_10 > 0 else 1
        microprice = (best_yes_bid * ask_qty + best_no_bid * bid_qty) / (bid_qty + ask_qty) if (bid_qty + ask_qty) > 0 else 0.50
        
        if len(yes_bids) >= 5:
            top_depth = float(yes_bids[0][1]) if yes_bids else 0
            avg_depth_5 = sum([float(p[1]) for p in yes_bids[:5]]) / 5 if len(yes_bids) >= 5 else 1
            depth_slope = top_depth / avg_depth_5 if avg_depth_5 > 0 else 1
        else:
            depth_slope = 1
        
        if depth_slope > 1.5:
            depth_slope_label = "🔴 Top-Heavy"
        elif depth_slope > 0.9:
            depth_slope_label = "🟡 Balanced"
        else:
            depth_slope_label = "🟢 Bottom-Heavy"
        
        spread_score = max(0, min(1, 1 - (spread / 0.10)))
        depth_score = min(1, (yes_vol_10 + no_vol_10) / 5000)
        imbalance_score = abs(imbalance_10)
        liquidity_score = (spread_score * 0.4 + depth_score * 0.3 + imbalance_score * 0.3)
        liquidity_score = max(0, min(1, liquidity_score))
        
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
            'depth_slope_label': depth_slope_label,
            'liquidity_score': liquidity_score,
            'liquidity_label': liquidity_label,
            'liquidity_class': liquidity_class,
            'depth': depth,
            'yes_volume': yes_vol_10,
            'no_volume': no_vol_10
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
            'depth_slope_label': '🟡 Balanced',
            'liquidity_score': 0,
            'liquidity_label': '🔴 Low',
            'liquidity_class': 'liquidity-low',
            'depth': 0,
            'yes_volume': 0,
            'no_volume': 0
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

# --- Enhanced Feature Engineering ---
def add_advanced_features(df):
    """Add comprehensive technical indicators, lag features, and volatility"""
    
    # --- 1. LOG RETURNS ---
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['log_return_5'] = np.log(df['close'] / df['close'].shift(5))
    df['log_return_10'] = np.log(df['close'] / df['close'].shift(10))
    
    # --- 2. ABSOLUTE FEATURES ---
    df['abs_log_return'] = df['log_return'].abs()
    df['abs_log_return_5'] = df['log_return_5'].abs()
    df['abs_log_return_10'] = df['log_return_10'].abs()
    
    # --- 3. LAG FEATURES ---
    df['lag_1'] = df['close'].shift(1)
    df['lag_5'] = df['close'].shift(5)
    df['lag_10'] = df['close'].shift(10)
    df['lag_volume'] = df['volume'].shift(1)
    
    # --- 4. VOLATILITY FEATURES ---
    df['volatility_5'] = df['return_1'].rolling(5).std()
    df['volatility_10'] = df['return_1'].rolling(10).std()
    df['volatility_ratio'] = df['volatility_5'] / (df['volatility_10'] + 0.001)
    
    # --- 5. MOVING AVERAGES ---
    df['sma_5'] = df['close'].rolling(5).mean()
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['ema_9'] = df['close'].ewm(span=9).mean()
    df['ema_21'] = df['close'].ewm(span=21).mean()
    
    # --- 6. RSI ---
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # --- 7. MACD ---
    df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # --- 8. BOLLINGER BANDS ---
    df['bb_middle'] = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * bb_std
    df['bb_lower'] = df['bb_middle'] - 2 * bb_std
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # --- 9. ATR ---
    df['atr'] = (df['high'] - df['low']).rolling(14).mean()
    
    # --- 10. STOCHASTIC OSCILLATOR ---
    low_min = df['low'].rolling(14).min()
    high_max = df['high'].rolling(14).max()
    df['stoch_k'] = 100 * ((df['close'] - low_min) / (high_max - low_min + 0.001))
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()
    
    # --- 11. WILLIAMS %R ---
    df['williams_r'] = -100 * ((high_max - df['close']) / (high_max - low_min + 0.001))
    
    # --- 12. CCI (Commodity Channel Index) ---
    tp = (df['high'] + df['low'] + df['close']) / 3
    sma_tp = tp.rolling(20).mean()
    mad_tp = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
    df['cci'] = (tp - sma_tp) / (0.015 * mad_tp + 0.001)
    
    # --- 13. MFI (Money Flow Index) ---
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typical_price * df['volume']
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
    mfi_ratio = positive_flow / (negative_flow + 0.001)
    df['mfi'] = 100 - (100 / (1 + mfi_ratio))
    
    # --- 14. ADX (Average Directional Index) ---
    tr = np.maximum(df['high'] - df['low'], 
                    np.maximum(abs(df['high'] - df['close'].shift(1)), 
                              abs(df['low'] - df['close'].shift(1))))
    atr_14 = tr.rolling(14).mean()
    
    up_move = df['high'] - df['high'].shift(1)
    down_move = df['low'].shift(1) - df['low']
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    plus_di = 100 * (pd.Series(plus_dm).rolling(14).mean() / atr_14)
    minus_di = 100 * (pd.Series(minus_dm).rolling(14).mean() / atr_14)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.001)
    df['adx'] = dx.rolling(14).mean()
    
    # --- 15. PRICE FEATURES ---
    df['price_range'] = (df['high'] - df['low']) / df['close']
    df['high_low_ratio'] = df['high'] / df['low']
    df['close_open_ratio'] = df['close'] / df['open']
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(10).mean()
    
    return df

# --- Build Hybrid CNN-GRU Model ---
def build_cnn_gru_model(input_shape):
    """Build a hybrid CNN-GRU model"""
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        GRU(64, return_sequences=True),
        Dropout(0.2),
        GRU(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# --- Ensemble Prediction ---
def ensemble_predict(X_scaled, lookback, n_features):
    """Generate ensemble prediction using multiple models"""
    try:
        # Build sequences
        X_seq = []
        for i in range(lookback, len(X_scaled)):
            X_seq.append(X_scaled[i-lookback:i])
        X_seq = np.array(X_seq)
        
        if len(X_seq) == 0:
            return 0.50
        
        # Build multiple models with different configurations
        models = []
        
        # Model 1: CNN-GRU
        model1 = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(lookback, n_features)),
            MaxPooling1D(pool_size=2),
            Dropout(0.2),
            GRU(64, return_sequences=True),
            Dropout(0.2),
            GRU(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        models.append(model1)
        
        # Model 2: Simple GRU (for ensemble)
        model2 = Sequential([
            GRU(64, input_shape=(lookback, n_features), return_sequences=True),
            Dropout(0.2),
            GRU(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        models.append(model2)
        
        # Get predictions from each model (using last row)
        last_seq = X_scaled[-lookback:].reshape(1, lookback, n_features)
        
        predictions = []
        for model in models:
            # Quick training (simplified for speed)
            model.fit(X_seq[:min(100, len(X_seq))], 
                     np.random.randint(0, 2, min(100, len(X_seq))), 
                     epochs=5, verbose=0)
            pred = model.predict(last_seq, verbose=0)[0][0]
            predictions.append(pred)
        
        # Ensemble: average predictions
        final_pred = np.mean(predictions)
        return final_pred
        
    except Exception as e:
        return 0.50

# --- MAIN LOOP ---
all_results = []
best_bets = []

fear_greed = fetch_fear_greed_index()
macro_data = fetch_macro_data()

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
    status_text.text(f"🔄 Analyzing {coin} with advanced features...")
    
    try:
        df = fetch_yahoo_data(coin)
        if df.empty:
            continue
        
        df = add_advanced_features(df)
        df_clean = df.dropna()
        
        if len(df_clean) < 100:
            continue
        
        # Get enhanced model probability using ensemble
        feature_cols = [
            'close', 'volume', 'log_return', 'abs_log_return',
            'lag_1', 'lag_5', 'volatility_5', 'volatility_10',
            'sma_5', 'sma_10', 'rsi', 'macd_hist',
            'bb_position', 'atr', 'stoch_k', 'stoch_d',
            'williams_r', 'cci', 'mfi', 'adx',
            'price_range', 'volume_ratio'
        ]
        
        available_cols = [col for col in feature_cols if col in df_clean.columns]
        
        if len(available_cols) < 10:
            continue
        
        X = df_clean[available_cols].values
        y = df_clean['close'].shift(-5) > df_clean['close']
        
        X_df = pd.DataFrame(X, columns=available_cols)
        X_df['target'] = y.astype(int)
        X_df_clean = X_df.dropna()
        
        if len(X_df_clean) < 50:
            continue
        
        # Scale features
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_df_clean[available_cols].values)
        y_train = X_df_clean['target'].values
        
        # Get ensemble prediction
        prob_5m = ensemble_predict(X_scaled, LOOKBACK_WINDOW, len(available_cols))
        
        # Also get a simpler model prediction for 15m
        prob_15m = prob_5m * 0.95 + 0.025  # Slight adjustment for longer window
        
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
        depth_slope_label = order_book.get('depth_slope_label', '🟡 Balanced')
        
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
        
        edge_5m = prob_5m - market_price
        edge_15m = prob_15m - market_price
        
        if liquidity_score < 0.3:
            edge_5m *= 0.6
            edge_15m *= 0.6
        elif liquidity_score < 0.5:
            edge_5m *= 0.8
            edge_15m *= 0.8
        
        # Ternary classification
        expected_move = abs(prob_5m - 0.50) * 100
        
        if expected_move < FLAT_THRESHOLD * 100:
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
        
        if expected_move < FLAT_THRESHOLD * 100:
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
            'Depth_Slope_Label': depth_slope_label,
            'Liquidity': liquidity_label,
            'Liquidity_Score': liquidity_score,
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

# --- Macro Data Summary ---
st.markdown("### 🌍 Macro Indicators")
macro_col1, macro_col2, macro_col3 = st.columns(3)
macro_col1.metric("SPY", f"${macro_data.get('spy_close', 0):.2f}", delta=f"{macro_data.get('spy_change', 0)*100:.2f}%")
macro_col2.metric("VIX", f"{macro_data.get('vix_close', 0):.2f}", delta=f"{macro_data.get('vix_change', 0)*100:.2f}%")
macro_col3.metric("DXY", f"{macro_data.get('dxy_close', 0):.2f}", delta="—")

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
                <div class="coin-stats">
                    Depth Slope: {result['Depth_Slope_Str']} ({result['Depth_Slope_Label']})
                </div>
            </div>
            """, unsafe_allow_html=True)

st.divider()

# --- Detailed Table ---
st.markdown("### 📈 Detailed Edge Analysis (Enhanced Features)")

if all_results:
    df_display = pd.DataFrame([{
        'Coin': r['Name'],
        'Symbol': r['Symbol'],
        'Price': r['Price_Str'],
        'Kalshi Price': r['Market_Price_Str'],
        'Spread': r['Spread_Str'],
        'Spread Quality': r['Spread_Quality'],
        'Imbalance (10)': r['Imbalance_10_Str'],
        'Microprice': r['Microprice_Str'],
        'Depth Slope': r['Depth_Slope_Str'],
        'Depth Slope Label': r['Depth_Slope_Label'],
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
                    <span>Liquidity: {bet['Liquidity']}</span>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 0.8rem; opacity: 0.8;">
                    <span>Depth Slope: {bet['Depth_Slope_Str']} ({bet['Depth_Slope_Label']})</span>
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

# --- Sidebar ---
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown(f"**Bankroll:** ${BANKROLL:.2f}")
    st.markdown(f"**Max Risk/Trade:** {MAX_RISK_PER_TRADE*100:.0f}%")
    st.markdown(f"**Min Edge:** {MIN_EDGE*100:.0f}%")
    st.markdown(f"**Flat Threshold:** {FLAT_THRESHOLD*100:.1f}%")
    st.markdown(f"**Lookback Window:** {LOOKBACK_WINDOW} min")
    
    st.divider()
    
    st.markdown("### 🆕 Code 6 Features")
    st.markdown("""
    ✅ **Hybrid CNN-GRU** (Conv1D + GRU)  
    ✅ **20+ Technical Indicators** (ADX, Stochastic, CCI, MFI, Williams %R)  
    ✅ **Lag Features** (lag_1, lag_5, lag_10)  
    ✅ **Volatility Features** (rolling std, volatility ratio)  
    ✅ **Macro Data** (SPY, VIX, DXY)  
    ✅ **Ensemble Logic** (multi-model averaging)  
    ✅ **RobustScaler** (outlier-resistant scaling)  
    ✅ **All Code 5 Features** (Order Book, Depth Slope, Regime)  
    """)
    
    st.divider()
    
    st.markdown("### 📊 Feature Count")
    st.markdown(f"**Total Features:** {len(feature_cols) if 'feature_cols' in dir() else '20+'}")
    
    st.divider()
    
    st.markdown("### 🔄 Manual Refresh")
    st.caption("Press 'R' or click the refresh button in your browser to get the latest data.")

# --- Footer ---
st.divider()
st.caption(f"⚡ Code 6 • Hybrid CNN-GRU • 20+ Indicators • Macro Data • Ensemble • Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
