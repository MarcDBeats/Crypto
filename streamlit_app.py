# ============================================
# CODE 8: KALSHI PRICE PREDICTOR WITH 5 & 10 MINUTE PREDICTIONS
# Features: Full data model + Limit Analysis
#           Separate 5-minute and 10-minute predictions
#           Compare confidence and direction across time horizons
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')
import pytz

# --- TIME ZONE SETUP ---
LOCAL_TZ = pytz.timezone('America/Chicago')

def get_current_ct_time():
    return datetime.now(LOCAL_TZ)

def get_next_kalshi_settlement():
    now = get_current_ct_time()
    minute = now.minute
    seconds = now.second
    minutes_to_next = (15 - (minute % 15)) % 15
    if minutes_to_next == 0 and seconds == 0:
        minutes_to_next = 15
    next_settlement = now + timedelta(minutes=minutes_to_next)
    next_settlement = next_settlement.replace(second=0, microsecond=0)
    return next_settlement, minutes_to_next

# --- Page Config ---
st.set_page_config(
    page_title="Kalshi Price Predictor",
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
    .price-card {
        background: linear-gradient(135deg, #1e1e2f 0%, #2d2d44 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #3d3d5c;
        text-align: center;
        min-height: 160px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .price-card .coin-name {
        font-size: 1rem;
        font-weight: 700;
        color: #ccc;
    }
    .price-card .price {
        font-size: 1.6rem;
        font-weight: 700;
        margin: 0.2rem 0;
        color: #fff;
    }
    .price-card .direction {
        font-size: 1.1rem;
        font-weight: 600;
        margin: 0.1rem 0;
    }
    .price-card .confidence-bar {
        height: 3px;
        border-radius: 2px;
        margin-top: 0.2rem;
        background: #2d2d44;
    }
    .price-card .confidence-fill {
        height: 100%;
        border-radius: 2px;
        transition: width 0.5s;
    }
    .price-card .info-row {
        display: flex;
        justify-content: space-between;
        font-size: 0.6rem;
        color: #888;
        margin-top: 0.1rem;
    }
    .direction-up {
        color: #00b894;
    }
    .direction-down {
        color: #ff6b6b;
    }
    .direction-wait {
        color: #fdcb6e;
    }
    .countdown {
        text-align: center;
        padding: 1rem;
        border-radius: 0.5rem;
        background: linear-gradient(135deg, #1a1a2e 0%, #2d2d44 100%);
        border: 2px solid #3d3d5c;
        margin-bottom: 1rem;
    }
    .countdown .timer {
        font-size: 2.5rem;
        font-weight: 700;
        color: #fdcb6e;
    }
    .countdown .label {
        font-size: 1rem;
        color: #888;
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
    .active-window {
        border: 2px solid #00b894;
        box-shadow: 0 0 20px rgba(0, 184, 148, 0.3);
    }
    .inactive-window {
        border: 1px solid #3d3d5c;
        opacity: 0.6;
    }
    .summary-bar {
        display: flex;
        justify-content: space-around;
        padding: 0.5rem;
        background: #1a1a2e;
        border-radius: 0.5rem;
        border: 1px solid #3d3d5c;
        margin-bottom: 1rem;
    }
    .summary-item {
        text-align: center;
    }
    .summary-item .number {
        font-size: 1.5rem;
        font-weight: 700;
    }
    .summary-item .label {
        font-size: 0.7rem;
        color: #888;
    }
    .limit-box {
        background: #0d0d1a;
        padding: 0.3rem;
        border-radius: 0.3rem;
        border: 1px solid #2d2d44;
        margin-top: 0.2rem;
        font-size: 0.6rem;
    }
    .limit-box .label {
        color: #888;
    }
    .limit-box .value {
        font-weight: 600;
    }
    .limit-box .stable {
        color: #00b894;
    }
    .limit-box .unstable {
        color: #fdcb6e;
    }
    .limit-box .volatile {
        color: #ff6b6b;
    }
    .ct-badge {
        background: #fdcb6e33;
        color: #fdcb6e;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        font-size: 0.7rem;
        font-weight: 600;
        display: inline-block;
    }
    .timeframe-badge {
        font-size: 0.6rem;
        background: #2d2d44;
        padding: 0.1rem 0.3rem;
        border-radius: 0.2rem;
        color: #888;
        margin: 0.1rem;
    }
    .prediction-comparison {
        display: flex;
        gap: 0.3rem;
        justify-content: center;
        font-size: 0.6rem;
        margin-top: 0.2rem;
    }
    .prediction-comparison .tf-5 {
        color: #667eea;
        font-weight: 600;
    }
    .prediction-comparison .tf-10 {
        color: #fdcb6e;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
current_ct = get_current_ct_time()
st.markdown('<div class="main-header">📊 Kalshi Price Predictor</div>', unsafe_allow_html=True)
st.caption(f"⚡ 5-min & 10-min predictions • Limit Analysis • All times in CT • Updated: {current_ct.strftime('%Y-%m-%d %H:%M:%S')} CT")

# --- Settings ---
BANKROLL = 100.00
MAX_RISK_PER_TRADE = 0.02
MIN_EDGE = 0.05
FLAT_THRESHOLD = 0.002
PREDICT_WINDOW = 5
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

# --- Helper Functions ---
def fmt_price(val):
    if val is None or pd.isna(val):
        return '—'
    return f"${val:,.2f}"

def fmt_confidence(val):
    if val is None or pd.isna(val):
        return '—'
    return f"{val:.0%}"

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

@st.cache_data(ttl=5)
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
        elif spread < 0.05:
            spread_quality = "🟡 Good"
        else:
            spread_quality = "🔴 Poor"
        
        yes_vol_10 = sum([float(p[1]) for p in yes_bids[:10]]) if yes_bids else 0
        no_vol_10 = sum([float(p[1]) for p in no_bids[:10]]) if no_bids else 0
        imbalance_10 = (yes_vol_10 - no_vol_10) / (yes_vol_10 + no_vol_10 + 1) if (yes_vol_10 + no_vol_10) > 0 else 0
        
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
        elif liquidity_score >= 0.4:
            liquidity_label = "🟡 Medium"
        else:
            liquidity_label = "🔴 Low"
        
        return {
            'best_yes_bid': best_yes_bid,
            'best_no_bid': best_no_bid,
            'spread': spread,
            'spread_quality': spread_quality,
            'imbalance_10': imbalance_10,
            'depth_slope': depth_slope,
            'depth_slope_label': depth_slope_label,
            'liquidity_score': liquidity_score,
            'liquidity_label': liquidity_label,
            'yes_volume': yes_vol_10,
            'no_volume': no_vol_10
        }
    except:
        return {
            'best_yes_bid': 0,
            'best_no_bid': 0,
            'spread': 0,
            'spread_quality': '🔴 Poor',
            'imbalance_10': 0,
            'depth_slope': 1,
            'depth_slope_label': '🟡 Balanced',
            'liquidity_score': 0,
            'liquidity_label': '🔴 Low',
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

# --- Feature Engineering ---
def add_advanced_features(df):
    """Add comprehensive technical indicators"""
    
    df['return_1'] = df['close'].pct_change()
    df['return_5'] = df['close'].pct_change(5)
    
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['abs_log_return'] = df['log_return'].abs()
    
    df['lag_1'] = df['close'].shift(1)
    df['lag_5'] = df['close'].shift(5)
    
    df['volatility_5'] = df['return_1'].rolling(5).std()
    df['volatility_10'] = df['return_1'].rolling(10).std()
    df['volatility_ratio'] = df['volatility_5'] / (df['volatility_10'] + 0.001)
    
    df['sma_5'] = df['close'].rolling(5).mean()
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_20'] = df['close'].rolling(20).mean()
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
    
    df['bb_middle'] = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * bb_std
    df['bb_lower'] = df['bb_middle'] - 2 * bb_std
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    df['atr'] = (df['high'] - df['low']).rolling(14).mean()
    
    low_min = df['low'].rolling(14).min()
    high_max = df['high'].rolling(14).max()
    df['stoch_k'] = 100 * ((df['close'] - low_min) / (high_max - low_min + 0.001))
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()
    
    df['williams_r'] = -100 * ((high_max - df['close']) / (high_max - low_min + 0.001))
    
    tp = (df['high'] + df['low'] + df['close']) / 3
    sma_tp = tp.rolling(20).mean()
    mad_tp = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
    df['cci'] = (tp - sma_tp) / (0.015 * mad_tp + 0.001)
    
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typical_price * df['volume']
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
    mfi_ratio = positive_flow / (negative_flow + 0.001)
    df['mfi'] = 100 - (100 / (1 + mfi_ratio))
    
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
    
    df['price_range'] = (df['high'] - df['low']) / df['close']
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(10).mean()
    
    return df

# --- Get Prediction for Specific Time Horizon ---
def get_prediction_for_horizon(coin_symbol, minutes_ahead, df_clean=None, feature_cols=None):
    try:
        if df_clean is None:
            df = fetch_yahoo_data(coin_symbol)
            if df.empty:
                return None
            df = add_advanced_features(df)
            df_clean = df.dropna()
        
        if len(df_clean) < 60:
            return None
        
        current_price = df_clean['close'].iloc[-1]
        
        if feature_cols is None:
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
            return None
        
        X = df_clean[available_cols].values
        y = df_clean['close'].shift(-minutes_ahead) > df_clean['close']
        
        X_df = pd.DataFrame(X, columns=available_cols)
        X_df['target'] = y.astype(int)
        X_df_clean = X_df.dropna()
        
        if len(X_df_clean) < 30:
            return None
        
        X_train = X_df_clean[available_cols].values[:-1]
        y_train = X_df_clean['target'].values[:-1]
        X_test = X_df_clean[available_cols].values[-1:].reshape(1, -1)
        
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        win_prob = model.predict_proba(X_test_scaled)[0][1]
        
        if win_prob > 0.5:
            target_price = current_price * (1 + (win_prob - 0.5) * 0.02)
        else:
            target_price = current_price * (1 - (0.5 - win_prob) * 0.02)
        
        return {
            'win_prob': win_prob,
            'target_price': target_price
        }
        
    except Exception as e:
        return None

# --- LIMIT ANALYSIS FUNCTION ---
def limit_analysis(coin_symbol, df_clean, feature_cols, minutes_until):
    try:
        available_cols = [col for col in feature_cols if col in df_clean.columns]
        if len(available_cols) < 10:
            return None
        
        window_size = min(5, len(df_clean) - 1)
        if window_size < 2:
            return None
        
        confidences = []
        timestamps = []
        
        for i in range(max(0, len(df_clean) - window_size - 1), len(df_clean) - 1):
            train_data = df_clean.iloc[:i+1]
            if len(train_data) < 30:
                continue
            
            X = train_data[available_cols].values
            y = train_data['close'].shift(-minutes_until) > train_data['close']
            
            X_df = pd.DataFrame(X, columns=available_cols)
            X_df['target'] = y.astype(int)
            X_df_clean = X_df.dropna()
            
            if len(X_df_clean) < 20:
                continue
            
            X_train = X_df_clean[available_cols].values[:-1]
            y_train = X_df_clean['target'].values[:-1]
            
            if len(X_train) < 10:
                continue
            
            try:
                model = RandomForestClassifier(n_estimators=20, max_depth=3, random_state=42)
                model.fit(X_train, y_train)
                
                X_test = X_df_clean[available_cols].values[-1:].reshape(1, -1)
                prob = model.predict_proba(X_test)[0][1]
                
                confidences.append(prob)
                timestamps.append(train_data['time'].iloc[-1])
            except:
                continue
        
        if len(confidences) < 2:
            return None
        
        time_diffs = []
        conf_diffs = []
        
        for i in range(1, len(confidences)):
            time_diff = (timestamps[i] - timestamps[i-1]).total_seconds() / 60
            conf_diff = confidences[i] - confidences[i-1]
            
            if time_diff > 0:
                time_diffs.append(time_diff)
                conf_diffs.append(conf_diff)
        
        if not time_diffs:
            return None
        
        avg_time_diff = np.mean(time_diffs)
        avg_conf_diff = np.mean(conf_diffs)
        rate_of_change = avg_conf_diff / avg_time_diff if avg_time_diff > 0 else 0
        
        conf_std = np.std(confidences) if len(confidences) > 1 else 0
        stability = max(0, 1 - min(1, conf_std * 2))
        
        last_conf = confidences[-1]
        
        if abs(rate_of_change) > 0.001:
            limit_estimate = last_conf + rate_of_change * minutes_until
        else:
            limit_estimate = last_conf
        
        limit_estimate = max(0.05, min(0.95, limit_estimate))
        
        if stability > 0.7 and abs(rate_of_change) < 0.01:
            stability_label = "🟢 Stable"
            stability_class = "stable"
        elif stability > 0.4 and abs(rate_of_change) < 0.03:
            stability_label = "🟡 Moderate"
            stability_class = "unstable"
        else:
            stability_label = "🔴 Volatile"
            stability_class = "volatile"
        
        return {
            'confidence_history': confidences,
            'timestamps': timestamps,
            'rate_of_change': rate_of_change,
            'stability': stability,
            'stability_label': stability_label,
            'stability_class': stability_class,
            'limit_estimate': limit_estimate,
            'last_confidence': last_conf,
            'trend': '📈 Increasing' if rate_of_change > 0.005 else '📉 Decreasing' if rate_of_change < -0.005 else '➡️ Flat'
        }
        
    except Exception as e:
        return None

# --- Get Full Prediction with All Timeframes ---
def get_full_prediction(coin_symbol):
    try:
        df = fetch_yahoo_data(coin_symbol)
        if df.empty:
            return None
        
        df = add_advanced_features(df)
        df_clean = df.dropna()
        
        if len(df_clean) < 60:
            return None
        
        current_price = df_clean['close'].iloc[-1]
        
        # Get minutes to next settlement
        _, minutes_until = get_next_kalshi_settlement()
        if minutes_until > PREDICT_WINDOW + 5:
            minutes_until = 15
        
        # Features
        feature_cols = [
            'close', 'volume', 'log_return', 'abs_log_return',
            'lag_1', 'lag_5', 'volatility_5', 'volatility_10',
            'sma_5', 'sma_10', 'rsi', 'macd_hist',
            'bb_position', 'atr', 'stoch_k', 'stoch_d',
            'williams_r', 'cci', 'mfi', 'adx',
            'price_range', 'volume_ratio'
        ]
        
        # --- PREDICTIONS FOR 5 AND 10 MINUTES ---
        pred_5 = get_prediction_for_horizon(coin_symbol, 5, df_clean, feature_cols)
        pred_10 = get_prediction_for_horizon(coin_symbol, 10, df_clean, feature_cols)
        
        # --- FETCH KALSHI DATA ---
        ticker = KALSHI_TICKERS.get(coin_symbol, '')
        kalshi_price = fetch_kalshi_price(ticker) if ticker else 0.50
        order_book = fetch_kalshi_order_book(ticker, depth=10) if ticker else {}
        
        # Order book data
        spread = order_book.get('spread', 0)
        spread_quality = order_book.get('spread_quality', '🔴 Poor')
        liquidity_score = order_book.get('liquidity_score', 0)
        liquidity_label = order_book.get('liquidity_label', '🔴 Low')
        imbalance_10 = order_book.get('imbalance_10', 0)
        depth_slope = order_book.get('depth_slope', 1)
        depth_slope_label = order_book.get('depth_slope_label', '🟡 Balanced')
        
        # Fear & Greed
        fear_greed = fetch_fear_greed_index()
        
        # --- LIMIT ANALYSIS (using 5-minute horizon for limit) ---
        limit_data = limit_analysis(coin_symbol, df_clean, feature_cols, 5)
        
        # --- Determine primary signal based on 5-minute prediction ---
        if pred_5:
            win_prob = pred_5['win_prob']
            edge = win_prob - kalshi_price
            
            if liquidity_score < 0.3:
                edge *= 0.6
            elif liquidity_score < 0.5:
                edge *= 0.8
            
            if edge >= MIN_EDGE and win_prob > 0.55:
                direction = "UP"
                signal = "BUY YES"
            elif edge <= -MIN_EDGE and win_prob < 0.45:
                direction = "DOWN"
                signal = "BUY NO"
            else:
                direction = "WAIT"
                signal = "SKIP"
        else:
            win_prob = 0.50
            direction = "WAIT"
            signal = "SKIP"
            edge = 0
        
        return {
            'current_price': current_price,
            'pred_5': pred_5,
            'pred_10': pred_10,
            'win_prob': win_prob,
            'edge': edge,
            'direction': direction,
            'signal': signal,
            'minutes_until': minutes_until,
            'kalshi_price': kalshi_price,
            'limit_data': limit_data,
            '_spread': spread,
            '_spread_quality': spread_quality,
            '_liquidity_score': liquidity_score,
            '_liquidity_label': liquidity_label,
            '_imbalance_10': imbalance_10,
            '_depth_slope': depth_slope,
            '_depth_slope_label': depth_slope_label,
            '_fear_greed': fear_greed
        }
        
    except Exception as e:
        return None

# --- MAIN LOOP ---
next_settlement, minutes_until = get_next_kalshi_settlement()
is_active = minutes_until <= PREDICT_WINDOW

# --- Countdown Timer ---
st.markdown(f"""
<div class="countdown">
    <div class="label">⏰ Next Settlement <span class="ct-badge">CT</span></div>
    <div class="timer">{minutes_until}m</div>
    <div class="label">{next_settlement.strftime('%I:%M %p')} CT</div>
    <div style="margin-top: 0.5rem; font-size: 1rem; color: {'#00b894' if is_active else '#636e72'};">
        {'🟢 Trading Window Open!' if is_active else '⏳ Waiting for trading window...'}
    </div>
</div>
""", unsafe_allow_html=True)

# --- Status Message ---
if is_active:
    st.success(f"🟢 Trading window is OPEN! {minutes_until} minutes until settlement (CT).")
    st.caption("Compare 5-minute and 10-minute predictions below.")
else:
    st.warning(f"⏳ Trading window closed. Next window opens in {minutes_until - PREDICT_WINDOW} minutes (CT).")
    st.caption("Predictions shown for reference only. 5-min vs 10-min comparison available.")

st.divider()

# --- Get Predictions for All Coins ---
all_results = []
best_bets = []

progress_bar = st.progress(0)
status_text = st.empty()

for idx, coin in enumerate(COINS):
    status_text.text(f"🔄 Analyzing {coin}...")
    
    pred = get_full_prediction(coin)
    if pred:
        metadata = COIN_METADATA.get(coin, {'name': coin.replace('-USD', ''), 'symbol': coin.replace('-USD', ''), 'color': '#ffffff'})
        
        # Get predictions
        win_prob_5 = pred['pred_5']['win_prob'] if pred['pred_5'] else 0.50
        target_5 = pred['pred_5']['target_price'] if pred['pred_5'] else pred['current_price']
        win_prob_10 = pred['pred_10']['win_prob'] if pred['pred_10'] else 0.50
        target_10 = pred['pred_10']['target_price'] if pred['pred_10'] else pred['current_price']
        
        # Determine if 5 and 10 minute predictions agree
        direction_5 = "UP" if win_prob_5 > 0.55 else "DOWN" if win_prob_5 < 0.45 else "WAIT"
        direction_10 = "UP" if win_prob_10 > 0.55 else "DOWN" if win_prob_10 < 0.45 else "WAIT"
        agree = direction_5 == direction_10 and direction_5 != "WAIT"
        
        result = {
            'Name': metadata['name'],
            'Symbol': metadata['symbol'],
            'Current_Price': pred['current_price'],
            'Current_Price_Str': fmt_price(pred['current_price']),
            'Win_Prob_5': win_prob_5,
            'Win_Prob_5_Str': fmt_confidence(win_prob_5),
            'Target_5_Str': fmt_price(target_5),
            'Win_Prob_10': win_prob_10,
            'Win_Prob_10_Str': fmt_confidence(win_prob_10),
            'Target_10_Str': fmt_price(target_10),
            'Direction_5': direction_5,
            'Direction_10': direction_10,
            'Agree': agree,
            'Signal': pred['signal'],
            'Edge': pred['edge'],
            'Edge_Str': f"{pred['edge']:.1%}",
            'Kalshi_Price_Str': f"{pred['kalshi_price']:.3f}",
            'Limit_Data': pred['limit_data'],
            'Color': metadata['color']
        }
        all_results.append(result)
        
        if pred['signal'] in ['BUY YES', 'BUY NO']:
            best_bets.append(result)
    
    progress_bar.progress((idx + 1) / len(COINS))

status_text.empty()
progress_bar.empty()

# --- Summary Bar ---
if all_results:
    signals_up = len([r for r in all_results if r['Signal'] == 'BUY YES'])
    signals_down = len([r for r in all_results if r['Signal'] == 'BUY NO'])
    signals_skip = len([r for r in all_results if r['Signal'] == 'SKIP'])
    agree_count = len([r for r in all_results if r['Agree']])
    
    st.markdown(f"""
    <div class="summary-bar">
        <div class="summary-item">
            <div class="number" style="color: #00b894;">{signals_up}</div>
            <div class="label">BUY YES</div>
        </div>
        <div class="summary-item">
            <div class="number" style="color: #ff6b6b;">{signals_down}</div>
            <div class="label">BUY NO</div>
        </div>
        <div class="summary-item">
            <div class="number" style="color: #636e72;">{signals_skip}</div>
            <div class="label">SKIP</div>
        </div>
        <div class="summary-item">
            <div class="number" style="color: #667eea;">{agree_count}</div>
            <div class="label">✅ 5 & 10 Agree</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# --- Display Price Predictions with 5 & 10 Minute Comparison ---
st.markdown("### 📊 Price Predictions for Next Settlement")
st.caption(f"Settlement at {next_settlement.strftime('%I:%M %p CT')} ({minutes_until}m remaining)")

cols = st.columns(len(COINS))

for i, result in enumerate(all_results):
    with cols[i]:
        # Use 5-minute direction for primary display
        if result['Direction_5'] == "UP":
            dir_class = "direction-up"
            dir_emoji = "⬆️"
        elif result['Direction_5'] == "DOWN":
            dir_class = "direction-down"
            dir_emoji = "⬇️"
        else:
            dir_class = "direction-wait"
            dir_emoji = "⏳"
        
        conf_pct = result['Win_Prob_5'] * 100
        conf_color = '#00b894' if conf_pct >= 65 else '#fdcb6e' if conf_pct >= 55 else '#ff6b6b'
        
        # Limit analysis display
        limit_html = ""
        if result['Limit_Data']:
            limit = result['Limit_Data']
            limit_est = limit['limit_estimate'] * 100
            stability_label = limit['stability_label']
            stability_class = limit['stability_class']
            
            limit_html = f"""
            <div class="limit-box">
                <div style="display: flex; justify-content: space-between;">
                    <span class="label">📈 Limit (est. accuracy):</span>
                    <span class="value" style="color: {conf_color};">{limit_est:.0f}%</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span class="label">🔄 Stability:</span>
                    <span class="value {stability_class}">{stability_label}</span>
                </div>
            </div>
            """
        
        # Agreement indicator
        agree_emoji = "✅" if result['Agree'] else "⚠️"
        agree_color = "#00b894" if result['Agree'] else "#fdcb6e"
        
        st.markdown(f"""
        <div class="price-card {'active-window' if is_active else 'inactive-window'}">
            <div class="coin-name">{result['Name']} ({result['Symbol']})</div>
            
            <div class="prediction-comparison">
                <span class="tf-5">5m: {result['Win_Prob_5_Str']} → {result['Target_5_Str']}</span>
                <span style="color:#888;">|</span>
                <span class="tf-10">10m: {result['Win_Prob_10_Str']} → {result['Target_10_Str']}</span>
            </div>
            
            <div class="direction {dir_class}">{dir_emoji} {result['Direction_5']}</div>
            
            <div class="info-row">
                <span>Current: {result['Current_Price_Str']}</span>
                <span>Kalshi: {result['Kalshi_Price_Str']}</span>
            </div>
            <div class="info-row">
                <span>Edge: {result['Edge_Str']}</span>
                <span>{agree_emoji} <span style="color:{agree_color};">{'Agree' if result['Agree'] else 'Disagree'}</span></span>
            </div>
            
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: {conf_pct:.0f}%; background: {conf_color};"></div>
            </div>
            
            {limit_html}
            
            <div style="font-size: 0.8rem; font-weight: 600; margin-top: 0.3rem; color: {'#00b894' if result['Signal'] == 'BUY YES' else '#ff6b6b' if result['Signal'] == 'BUY NO' else '#636e72'};">
                {result['Signal']}
            </div>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# --- Best Bets ---
if is_active and best_bets:
    st.markdown("### ⭐ Best Bets (Trading Window Open!)")
    st.caption("Highest confidence signals for immediate action")
    cols = st.columns(min(len(best_bets), 3))
    for i, bet in enumerate(best_bets[:3]):
        col = cols[i % 3]
        with col:
            if bet['Signal'] == "BUY YES":
                card_class = "best-bet"
                emoji = "🟢"
            else:
                card_class = "best-bet-no"
                emoji = "🔴"
            
            limit_est = ""
            if bet['Limit_Data']:
                limit_est = f"Est. Accuracy: {bet['Limit_Data']['limit_estimate']*100:.0f}%"
            
            agree_emoji = "✅" if bet['Agree'] else "⚠️"
            
            st.markdown(f"""
            <div class="{card_class}">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 1.2rem;">{bet['Name']} ({bet['Symbol']})</span>
                    <span style="font-size: 1.5rem;">{emoji} {bet['Direction_5']}</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-top: 0.5rem;">
                    <span>5m: {bet['Win_Prob_5_Str']}</span>
                    <span>10m: {bet['Win_Prob_10_Str']}</span>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 0.8rem; opacity: 0.8;">
                    <span>Edge: {bet['Edge_Str']}</span>
                    <span>{agree_emoji} {'Agree' if bet['Agree'] else 'Disagree'}</span>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 0.8rem; opacity: 0.8;">
                    <span>{limit_est}</span>
                    <span>Kalshi: {bet['Kalshi_Price_Str']}</span>
                </div>
                <div style="margin-top: 0.3rem; font-size: 0.8rem; font-weight: 600;">
                    {bet['Signal']}
                </div>
            </div>
            """, unsafe_allow_html=True)
elif not is_active and best_bets:
    st.info("⏳ Best bets available, but trading window is closed. Wait for the next window.")
else:
    st.markdown("""
    <div class="skip-bet">
        ⏳ No clear signals. Waiting for better opportunities...
    </div>
    """, unsafe_allow_html=True)

st.divider()

# --- How It Works ---
st.markdown("### 📖 How This Works")
st.markdown("""
**Time Zone**: All times displayed in **Central Time (CT)** — Texas time.

**Trading Window**: Only trades are shown when you're within **5 minutes** of the next Kalshi settlement.

**📊 5-Minute vs 10-Minute Predictions**:
- **5-Minute Prediction**: Closer to settlement — higher signal strength, more accurate
- **10-Minute Prediction**: Further out — shows the trend direction
- **Agreement**: If both timeframes agree on direction, it's a stronger signal
- **Disagreement**: If they disagree, be cautious — momentum may be shifting

**📈 Limit Analysis (Calculus-Inspired)**:
- **Limit (Est. Accuracy)**: The model's projected accuracy at the exact moment of settlement
- **Stability**: How steady the confidence is — stable = trustworthy, volatile = be careful

**Decisions**:
- 🟢 **BUY YES** → Model says price will go UP
- 🔴 **BUY NO** → Model says price will go DOWN
- ⏳ **SKIP** → No clear signal

**When to Trade**:
1. Wait for the 🟢 **Trading Window Open!** message
2. Look for ⭐ **Best Bets**
3. Check if **5 & 10 minute predictions agree** (✅)
4. Check the **Limit Analysis** — is confidence stable?
5. Follow the signal (BUY YES or BUY NO)
""")

st.divider()

# --- Sidebar ---
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown(f"**Time Zone:** Central Time (CT)")
    st.markdown(f"**Trading Window:** {PREDICT_WINDOW} minutes before settlement")
    st.markdown(f"**Min Edge:** {MIN_EDGE*100:.0f}%")
    st.markdown(f"**Current Window:** {'🟢 Open' if is_active else '🔴 Closed'}")
    st.markdown(f"**Minutes to Settlement:** {minutes_until}m")
    
    st.divider()
    
    st.markdown("### 📊 Timeframe Guide")
    st.markdown("""
    **5-Minute Prediction** (🔵 Blue):
    - Closer to settlement
    - Higher signal strength
    - More accurate
    
    **10-Minute Prediction** (🟡 Yellow):
    - Shows the trend
    - Earlier signal
    - More time for price to move
    
    **✅ Agree**: Both timeframes align → Strong signal
    **⚠️ Disagree**: Timeframes conflict → Be cautious
    """)
    
    st.divider()
    
    st.markdown("### 📈 Limit Analysis Guide")
    st.markdown("""
    **🟢 Stable**: Confidence is steady — trustworthy
    **🟡 Moderate**: Some fluctuation — use caution
    **🔴 Volatile**: Confidence is jumping — be careful
    """)
    
    st.divider()
    
    st.markdown("### 🔄 Refresh")
    st.caption("Click refresh in your browser to update data")

# --- Footer ---
st.caption(f"⚡ Code 8 • 5-min & 10-min predictions • Limit Analysis • Last updated: {get_current_ct_time().strftime('%Y-%m-%d %H:%M:%S')} CT")
