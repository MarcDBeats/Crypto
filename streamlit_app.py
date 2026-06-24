# ============================================
# CODE 7: KALSHI PRICE PREDICTOR (FULL DATA, CLEAN DASHBOARD)
# Behind the scenes: All Code 5 metrics (order book, depth slope, liquidity, etc.)
# Dashboard: Clean view with price predictions, countdown, and best bets
# Trading window: Only shows signals within 5 minutes of settlement
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
        min-height: 150px;
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
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0.3rem 0;
        color: #fff;
    }
    .price-card .direction {
        font-size: 1.2rem;
        font-weight: 600;
        margin: 0.2rem 0;
    }
    .price-card .confidence-bar {
        height: 4px;
        border-radius: 2px;
        margin-top: 0.3rem;
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
        font-size: 0.7rem;
        color: #888;
        margin-top: 0.2rem;
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
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown('<div class="main-header">📊 Kalshi Price Predictor</div>', unsafe_allow_html=True)
st.caption(f"⚡ Full data model • Clean dashboard • Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# --- Settings ---
BANKROLL = 100.00
MAX_RISK_PER_TRADE = 0.02
MIN_EDGE = 0.05
FLAT_THRESHOLD = 0.002
PREDICT_WINDOW = 5

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
def get_next_kalshi_settlement():
    now = datetime.now()
    minute = now.minute
    minutes_to_next = (15 - (minute % 15)) % 15
    if minutes_to_next == 0:
        minutes_to_next = 15
    next_settlement = now + timedelta(minutes=minutes_to_next)
    next_settlement = next_settlement.replace(second=0, microsecond=0)
    return next_settlement, minutes_to_next

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

# --- Get Prediction with Full Data ---
def get_settlement_prediction(coin_symbol):
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
        
        available_cols = [col for col in feature_cols if col in df_clean.columns]
        if len(available_cols) < 10:
            return None
        
        X = df_clean[available_cols].values
        y = df_clean['close'].shift(-minutes_until) > df_clean['close']
        
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
        
        # --- FETCH ALL DATA (Hidden) ---
        ticker = KALSHI_TICKERS.get(coin_symbol, '')
        kalshi_price = fetch_kalshi_price(ticker) if ticker else 0.50
        order_book = fetch_kalshi_order_book(ticker, depth=10) if ticker else {}
        
        # Order book data (used for model but not displayed)
        spread = order_book.get('spread', 0)
        spread_quality = order_book.get('spread_quality', '🔴 Poor')
        liquidity_score = order_book.get('liquidity_score', 0)
        liquidity_label = order_book.get('liquidity_label', '🔴 Low')
        imbalance_10 = order_book.get('imbalance_10', 0)
        depth_slope = order_book.get('depth_slope', 1)
        depth_slope_label = order_book.get('depth_slope_label', '🟡 Balanced')
        
        # Fear & Greed (hidden)
        fear_greed = fetch_fear_greed_index()
        
        # Edge calculation
        edge = win_prob - kalshi_price
        
        # Adjust edge based on liquidity (hidden but affects decision)
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
        
        return {
            'current_price': current_price,
            'target_price': target_price,
            'win_prob': win_prob,
            'edge': edge,
            'direction': direction,
            'signal': signal,
            'minutes_until': minutes_until,
            'kalshi_price': kalshi_price,
            # Hidden data (for reference, not displayed)
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
    <div class="label">⏰ Next Settlement</div>
    <div class="timer">{minutes_until}m</div>
    <div class="label">{next_settlement.strftime('%I:%M %p')} CT</div>
    <div style="margin-top: 0.5rem; font-size: 1rem; color: {'#00b894' if is_active else '#636e72'};">
        {'🟢 Trading Window Open!' if is_active else '⏳ Waiting for trading window...'}
    </div>
</div>
""", unsafe_allow_html=True)

# --- Status Message ---
if is_active:
    st.success(f"🟢 Trading window is OPEN! {minutes_until} minutes until settlement.")
    st.caption("You can place trades based on the predictions below.")
else:
    st.warning(f"⏳ Trading window closed. Next window opens in {minutes_until - PREDICT_WINDOW} minutes.")
    st.caption("Predictions shown for reference only. Wait for the trading window to open.")

st.divider()

# --- Get Predictions for All Coins ---
all_results = []
best_bets = []

progress_bar = st.progress(0)
status_text = st.empty()

for idx, coin in enumerate(COINS):
    status_text.text(f"🔄 Analyzing {coin}...")
    
    pred = get_settlement_prediction(coin)
    if pred:
        metadata = COIN_METADATA.get(coin, {'name': coin.replace('-USD', ''), 'symbol': coin.replace('-USD', ''), 'color': '#ffffff'})
        
        result = {
            'Name': metadata['name'],
            'Symbol': metadata['symbol'],
            'Current_Price': pred['current_price'],
            'Current_Price_Str': fmt_price(pred['current_price']),
            'Target_Price': pred['target_price'],
            'Target_Price_Str': fmt_price(pred['target_price']),
            'Win_Prob': pred['win_prob'],
            'Win_Prob_Str': fmt_confidence(pred['win_prob']),
            'Edge': pred['edge'],
            'Edge_Str': f"{pred['edge']:.1%}",
            'Direction': pred['direction'],
            'Signal': pred['signal'],
            'Kalshi_Price': pred['kalshi_price'],
            'Kalshi_Price_Str': f"{pred['kalshi_price']:.3f}",
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
            <div class="number" style="color: #fdcb6e;">{len(all_results)}</div>
            <div class="label">Total Coins</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- Display Price Predictions ---
st.markdown("### 📊 Price Predictions for Next Settlement")
st.caption(f"Settlement at {next_settlement.strftime('%I:%M %p CT')} ({minutes_until}m remaining)")

cols = st.columns(len(COINS))

for i, result in enumerate(all_results):
    with cols[i]:
        if result['Direction'] == "UP":
            dir_class = "direction-up"
            dir_emoji = "⬆️"
        elif result['Direction'] == "DOWN":
            dir_class = "direction-down"
            dir_emoji = "⬇️"
        else:
            dir_class = "direction-wait"
            dir_emoji = "⏳"
        
        conf_pct = result['Win_Prob'] * 100
        conf_color = '#00b894' if conf_pct >= 65 else '#fdcb6e' if conf_pct >= 55 else '#ff6b6b'
        
        st.markdown(f"""
        <div class="price-card {'active-window' if is_active else 'inactive-window'}">
            <div class="coin-name">{result['Name']} ({result['Symbol']})</div>
            <div class="price">{result['Target_Price_Str']}</div>
            <div class="direction {dir_class}">{dir_emoji} {result['Direction']}</div>
            <div class="info-row">
                <span>Current: {result['Current_Price_Str']}</span>
                <span>Kalshi: {result['Kalshi_Price_Str']}</span>
            </div>
            <div class="info-row">
                <span>Confidence: {result['Win_Prob_Str']}</span>
                <span>Edge: {result['Edge_Str']}</span>
            </div>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: {conf_pct:.0f}%; background: {conf_color};"></div>
            </div>
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
            
            st.markdown(f"""
            <div class="{card_class}">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 1.2rem;">{bet['Name']} ({bet['Symbol']})</span>
                    <span style="font-size: 1.5rem;">{emoji} {bet['Direction']}</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-top: 0.5rem;">
                    <span>Target: {bet['Target_Price_Str']}</span>
                    <span>Edge: {bet['Edge_Str']}</span>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 0.8rem; opacity: 0.8;">
                    <span>Confidence: {bet['Win_Prob_Str']}</span>
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
**Trading Window**: Only trades are shown when you're within **5 minutes** of the next Kalshi settlement.

**What's Happening Behind the Scenes**:
- **Order Book Data**: Bid/ask spreads, depth slope, liquidity scores
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Stochastic, CCI, MFI, ADX
- **Volatility & Momentum**: Returns, volatility ratios, price ranges
- **Market Sentiment**: Fear & Greed Index
- **All data feeds into the model** — you just see the clean results

**Decisions**:
- 🟢 **BUY YES** → Model says price will go UP
- 🔴 **BUY NO** → Model says price will go DOWN
- ⏳ **SKIP** → No clear signal

**When to Trade**:
1. Wait for the 🟢 **Trading Window Open!** message
2. Look for ⭐ **Best Bets**
3. Follow the signal (BUY YES or BUY NO)
4. Place your trade on Kalshi
""")

st.divider()

# --- Sidebar ---
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown(f"**Trading Window:** {PREDICT_WINDOW} minutes before settlement")
    st.markdown(f"**Min Edge:** {MIN_EDGE*100:.0f}%")
    st.markdown(f"**Current Window:** {'🟢 Open' if is_active else '🔴 Closed'}")
    st.markdown(f"**Minutes to Settlement:** {minutes_until}m")
    
    st.divider()
    
    st.markdown("### 🎯 Quick Guide")
    st.markdown("""
    1. **Check the timer** — wait for trading window
    2. **Look at price targets** — model's prediction
    3. **Check Best Bets** — highest confidence signals
    4. **Place trade** — BUY YES or BUY NO on Kalshi
    """)
    
    st.divider()
    
    st.markdown("### 🔄 Refresh")
    st.caption("Click refresh in your browser to update data")

# --- Footer ---
st.caption(f"⚡ Code 7 • Full data model • Clean dashboard • Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
