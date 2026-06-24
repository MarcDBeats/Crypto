# ============================================
# CODE 5.3: KALSHI EDGE DETECTOR WITH SETTLEMENT BACKTEST (FIXED)
# Features: All previous features + Time zone fix + Kalshi schedule alignment
#           All times displayed in Central Time (CT)
#           Settlement times exactly at :00, :15, :30, :45
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
import json
import os

# --- TIME ZONE SETUP ---
# Set your local time zone (Central Time)
LOCAL_TZ = 'America/Chicago'  # Texas Central Time

try:
    import pytz
    TZ_AVAILABLE = True
except ImportError:
    TZ_AVAILABLE = False
    st.warning("pytz not installed. Using UTC times. Run: pip install pytz")

def to_local_time(utc_dt):
    """Convert UTC datetime to Central Time"""
    if not TZ_AVAILABLE:
        return utc_dt
    if utc_dt.tzinfo is None:
        utc_dt = pytz.UTC.localize(utc_dt)
    return utc_dt.astimezone(pytz.timezone(LOCAL_TZ))

def get_current_local_time():
    """Get current Central Time"""
    if TZ_AVAILABLE:
        return datetime.now(pytz.timezone(LOCAL_TZ))
    else:
        return datetime.now()

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
        min-height: 180px;
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
        font-size: 1.2rem;
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
    .contract-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #2d2d44 100%);
        padding: 0.8rem;
        border-radius: 0.5rem;
        border: 1px solid #3d3d5c;
        text-align: center;
        margin: 0.2rem 0;
    }
    .contract-card .time-label {
        font-size: 0.8rem;
        color: #888;
    }
    .contract-card .direction {
        font-size: 1.2rem;
        font-weight: 700;
    }
    .contract-card .target-price {
        font-size: 1rem;
        font-weight: 600;
        color: #ccc;
    }
    .contract-card .confidence-bar {
        height: 4px;
        border-radius: 2px;
        margin-top: 0.3rem;
        background: #2d2d44;
    }
    .contract-card .confidence-fill {
        height: 100%;
        border-radius: 2px;
        transition: width 0.5s;
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
    .direction-flat {
        color: #636e72;
    }
    .settlement-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #2d2d44 100%);
        padding: 0.8rem;
        border-radius: 0.5rem;
        border: 2px solid #3d3d5c;
        text-align: center;
        margin: 0.2rem 0;
    }
    .settlement-win {
        border-color: #00b894 !important;
    }
    .settlement-loss {
        border-color: #ff6b6b !important;
    }
    .settlement-card .result-label {
        font-size: 0.8rem;
        font-weight: 700;
    }
    .settlement-card .result-win {
        color: #00b894;
    }
    .settlement-card .result-loss {
        color: #ff6b6b;
    }
    .settlement-card .pnl {
        font-size: 1.1rem;
        font-weight: 700;
    }
    .local-time {
        color: #fdcb6e;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
local_time = get_current_local_time()
st.markdown('<div class="main-header">📊 Kalshi Edge Detector Pro</div>', unsafe_allow_html=True)
st.caption(f"⚡ Code 5.3 • Settlement Backtest • All times in CT • Updated: {local_time.strftime('%Y-%m-%d %H:%M:%S')} CT")

# --- Settings ---
BANKROLL = 100.00
MAX_RISK_PER_TRADE = 0.02
MIN_EDGE = 0.05
FLAT_THRESHOLD = 0.002
BET_SIZE = 10.00

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

# --- SETTLEMENT TRACKER ---
class SettlementTracker:
    """Track previous settlements and compare model predictions"""
    
    def __init__(self, filename='settlement_data.json'):
        self.filename = filename
        self.data = self._load_data()
    
    def _load_data(self):
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r') as f:
                    return json.load(f)
            except:
                return {'history': [], 'stats': {}}
        return {'history': [], 'stats': {}}
    
    def _save_data(self):
        with open(self.filename, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def add_settlement(self, coin, settlement_time, predicted_direction, 
                       predicted_price, actual_price, confidence, edge,
                       win, pnl):
        """Record a settlement comparison"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'settlement_time': settlement_time.isoformat(),
            'coin': coin,
            'predicted_direction': predicted_direction,
            'predicted_price': predicted_price,
            'actual_price': actual_price,
            'confidence': confidence,
            'edge': edge,
            'win': win,
            'pnl': pnl
        }
        
        self.data['history'].append(entry)
        
        if len(self.data['history']) > 100:
            self.data['history'] = self.data['history'][-100:]
        
        self._update_stats()
        self._save_data()
    
    def _update_stats(self):
        history = self.data['history']
        if not history:
            return
        
        wins = sum(1 for h in history if h['win'])
        losses = len(history) - wins
        total_pnl = sum(h['pnl'] for h in history)
        
        coin_stats = {}
        for h in history:
            coin = h['coin']
            if coin not in coin_stats:
                coin_stats[coin] = {'wins': 0, 'losses': 0, 'pnl': 0}
            if h['win']:
                coin_stats[coin]['wins'] += 1
            else:
                coin_stats[coin]['losses'] += 1
            coin_stats[coin]['pnl'] += h['pnl']
        
        for coin in coin_stats:
            total = coin_stats[coin]['wins'] + coin_stats[coin]['losses']
            coin_stats[coin]['win_rate'] = coin_stats[coin]['wins'] / total if total > 0 else 0
        
        self.data['stats'] = {
            'total_trades': len(history),
            'wins': wins,
            'losses': losses,
            'win_rate': wins / len(history) if history else 0,
            'total_pnl': total_pnl,
            'avg_pnl': total_pnl / len(history) if history else 0,
            'coin_stats': coin_stats
        }
    
    def get_last_settlement(self, coin):
        history = self.data['history']
        for h in reversed(history):
            if h['coin'] == coin:
                return h
        return None
    
    def get_stats(self):
        return self.data['stats']

# Initialize settlement tracker
settlement_tracker = SettlementTracker()

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

# --- GET NEXT KALSHI SETTLEMENT TIME (FIXED) ---
def get_next_kalshi_settlement(current_time):
    """Get the next Kalshi settlement time (:00, :15, :30, :45)"""
    minute = current_time.minute
    # Round up to next :00, :15, :30, :45
    minutes_to_next = (15 - (minute % 15)) % 15
    if minutes_to_next == 0:
        minutes_to_next = 15  # If exactly on a settlement, go to next one
    
    next_settlement = current_time + timedelta(minutes=minutes_to_next)
    next_settlement = next_settlement.replace(second=0, microsecond=0)
    
    return next_settlement, minutes_to_next

# --- Get previous Kalshi settlement ---
def get_previous_kalshi_settlement(current_time):
    """Get the previous Kalshi settlement time (:00, :15, :30, :45)"""
    minute = current_time.minute
    # Round down to previous :00, :15, :30, :45
    minutes_since = minute % 15
    if minutes_since == 0:
        minutes_since = 15  # If exactly on a settlement, go back one
    
    prev_settlement = current_time - timedelta(minutes=minutes_since)
    prev_settlement = prev_settlement.replace(second=0, microsecond=0)
    
    return prev_settlement

# --- Get contract target with Kalshi alignment ---
def get_next_contract_target(df_clean, coin_symbol):
    """Get the model's price target for the next Kalshi contract"""
    try:
        current_price = df_clean['close'].iloc[-1]
        
        # --- FIX: Use local time for Kalshi settlement alignment ---
        if TZ_AVAILABLE:
            current_time = datetime.now(pytz.timezone(LOCAL_TZ))
        else:
            current_time = datetime.now()
        
        # Get next Kalshi settlement time
        next_settlement, minutes_until = get_next_kalshi_settlement(current_time)
        
        # If less than 2 minutes, use the next one
        if minutes_until < 2:
            next_settlement, minutes_until = get_next_kalshi_settlement(
                current_time + timedelta(minutes=15)
            )
        
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
        
        if len(X_df_clean) < 50:
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
        
        projected_move_pct = ((target_price - current_price) / current_price) * 100
        
        edge = win_prob - 0.50
        if edge >= MIN_EDGE and win_prob > 0.55:
            signal = "BUY YES"
            direction = "UP"
        elif edge <= -MIN_EDGE and win_prob < 0.45:
            signal = "BUY NO"
            direction = "DOWN"
        else:
            signal = "SKIP"
            direction = "WAIT"
        
        return {
            'current_price': current_price,
            'target_price': target_price,
            'projected_move_pct': projected_move_pct,
            'win_prob': win_prob,
            'edge': edge,
            'signal': signal,
            'direction': direction,
            'next_settlement': next_settlement,
            'minutes_until': minutes_until
        }
        
    except Exception as e:
        return None

# --- MAIN LOOP ---
all_results = []
best_bets = []
settlement_results = []

fear_greed = fetch_fear_greed_index()
macro_data = fetch_macro_data()

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

# --- Get contract targets ---
contract_targets = {}

for idx, coin in enumerate(COINS):
    status_text.text(f"🔄 Analyzing {coin} contract targets...")
    
    try:
        df = fetch_yahoo_data(coin)
        if df.empty:
            continue
        
        df = add_advanced_features(df)
        df_clean = df.dropna()
        
        if len(df_clean) < 100:
            continue
        
        target = get_next_contract_target(df_clean, coin)
        if target:
            contract_targets[coin] = target
            
    except Exception as e:
        pass
    
    progress_bar.progress((idx + 1) / len(COINS))

# --- Run regular dashboard and compare with previous settlement ---
for idx, coin in enumerate(COINS):
    status_text.text(f"🔄 Analyzing {coin}...")
    
    try:
        df = fetch_yahoo_data(coin)
        if df.empty:
            continue
        
        df = add_advanced_features(df)
        df_clean = df.dropna()
        
        if len(df_clean) < 100:
            continue
        
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
        y = df_clean['close'].shift(-15) > df_clean['close']
        
        X_df = pd.DataFrame(X, columns=available_cols)
        X_df['target'] = y.astype(int)
        X_df_clean = X_df.dropna()
        
        if len(X_df_clean) < 50:
            continue
        
        X_train = X_df_clean[available_cols].values[:-1]
        y_train = X_df_clean['target'].values[:-1]
        X_test = X_df_clean[available_cols].values[-1:].reshape(1, -1)
        
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        prob_5m = model.predict_proba(X_test_scaled)[0][1]
        prob_15m = prob_5m * 0.95 + 0.025
        
        market_price = kalshi_prices.get(coin, 0.50)
        
        order_book = kalshi_order_books.get(coin, {})
        spread = order_book.get('spread', 0)
        spread_quality = order_book.get('spread_quality', '🔴 Poor')
        liquidity_score = order_book.get('liquidity_score', 0)
        liquidity_label = order_book.get('liquidity_label', '🔴 Low')
        imbalance_10 = order_book.get('imbalance_10', 0)
        depth_slope = order_book.get('depth_slope', 1)
        depth_slope_label = order_book.get('depth_slope_label', '🟡 Balanced')
        
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
        
        # --- SIMULATE SETTLEMENT COMPARISON USING KALSHI SCHEDULE ---
        if len(df_clean) > 30:
            # Get current local time
            if TZ_AVAILABLE:
                current_local = datetime.now(pytz.timezone(LOCAL_TZ))
            else:
                current_local = datetime.now()
            
            # Get previous Kalshi settlement time
            prev_settlement = get_previous_kalshi_settlement(current_local)
            
            # Find the data point closest to the previous settlement time
            # (The actual price at that settlement time)
            closest_idx = None
            closest_diff = None
            
            for i in range(len(df_clean) - 1, -1, -1):
                row_time = df_clean['time'].iloc[i]
                # Convert row time to local time for comparison
                if TZ_AVAILABLE:
                    row_time_local = to_local_time(row_time)
                else:
                    row_time_local = row_time
                
                diff = abs((row_time_local - prev_settlement).total_seconds())
                if closest_diff is None or diff < closest_diff:
                    closest_diff = diff
                    closest_idx = i
                    # If within 1 minute, break
                    if diff < 60:
                        break
            
            if closest_idx is not None and closest_idx > 0:
                settlement_price = df_clean['close'].iloc[closest_idx]
                settlement_time = df_clean['time'].iloc[closest_idx]
                
                # Get the price 15 minutes before settlement (where the model made its prediction)
                pred_idx = closest_idx - 15
                if pred_idx > 0:
                    pred_price = df_clean['close'].iloc[pred_idx]
                    
                    # Get the model's prediction at that time
                    if len(df_clean) > pred_idx + 10:
                        try:
                            # Get data up to the prediction point
                            X_prev = df_clean[available_cols].iloc[:pred_idx].values
                            if len(X_prev) >= 20:
                                X_prev_last = X_prev[-1:].reshape(1, -1)
                                X_prev_scaled = scaler.transform(X_prev_last)
                                prev_prob = model.predict_proba(X_prev_scaled)[0][1]
                                
                                if prev_prob > 0.55:
                                    prev_direction = "UP"
                                elif prev_prob < 0.45:
                                    prev_direction = "DOWN"
                                else:
                                    prev_direction = "WAIT"
                                
                                prev_predicted_price = pred_price * (1 + (prev_prob - 0.5) * 0.02)
                                
                                actual_change_pct = ((settlement_price - pred_price) / pred_price) * 100
                                if prev_direction == "UP" and actual_change_pct > 0:
                                    win = True
                                    pnl = BET_SIZE * 0.95
                                elif prev_direction == "DOWN" and actual_change_pct < 0:
                                    win = True
                                    pnl = BET_SIZE * 0.95
                                elif prev_direction != "WAIT":
                                    win = False
                                    pnl = -BET_SIZE
                                else:
                                    win = None
                                    pnl = 0
                                
                                if prev_direction != "WAIT" and win is not None:
                                    settlement_tracker.add_settlement(
                                        coin=coin,
                                        settlement_time=settlement_time,
                                        predicted_direction=prev_direction,
                                        predicted_price=prev_predicted_price,
                                        actual_price=settlement_price,
                                        confidence=prev_prob,
                                        edge=prev_prob - 0.50,
                                        win=win,
                                        pnl=pnl
                                    )
                                    
                                    # Format time in CT
                                    if TZ_AVAILABLE:
                                        time_str = to_local_time(settlement_time).strftime('%I:%M %p CT')
                                    else:
                                        time_str = settlement_time.strftime('%I:%M %p CT')
                                    
                                    settlement_results.append({
                                        'Coin': metadata['name'],
                                        'Symbol': metadata['symbol'],
                                        'Predicted': prev_direction,
                                        'Predicted Price': f"${prev_predicted_price:.2f}",
                                        'Actual Price': f"${settlement_price:.2f}",
                                        'Result': '✅ WIN' if win else '❌ LOSS',
                                        'P&L': f"${pnl:.2f}",
                                        'Confidence': f"{prev_prob:.0%}",
                                        'Time': time_str
                                    })
                        except:
                            pass
        
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
            'Imbalance_10': imbalance_10,
            'Imbalance_10_Str': f"{imbalance_10:.2f}",
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

# --- Previous Settlement Results Section ---
st.markdown("### 📊 Previous Settlement Results")
st.caption(f"What would have happened if you placed a $10 bet on the model's last prediction (All times in CT)")

if settlement_results:
    df_settlement = pd.DataFrame(settlement_results)
    
    def color_result(val):
        if 'WIN' in str(val):
            return 'color: #00b894; font-weight: bold'
        elif 'LOSS' in str(val):
            return 'color: #ff6b6b; font-weight: bold'
        return ''
    
    def color_pnl(val):
        if isinstance(val, str) and '$' in val:
            try:
                num = float(val.replace('$', '').replace(',', ''))
                if num > 0:
                    return 'color: #00b894; font-weight: bold'
                elif num < 0:
                    return 'color: #ff6b6b; font-weight: bold'
            except:
                pass
        return ''
    
    try:
        styled_settlement = df_settlement.style.map(color_result, subset=['Result'])
        styled_settlement = styled_settlement.map(color_pnl, subset=['P&L'])
    except AttributeError:
        styled_settlement = df_settlement.style.applymap(color_result, subset=['Result'])
        styled_settlement = styled_settlement.applymap(color_pnl, subset=['P&L'])
    
    st.dataframe(styled_settlement, use_container_width=True, hide_index=True)
    
    stats = settlement_tracker.get_stats()
    if stats:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Trades", stats.get('total_trades', 0))
        col2.metric("Win Rate", f"{stats.get('win_rate', 0)*100:.0f}%")
        col3.metric("Total P&L", f"${stats.get('total_pnl', 0):.2f}", 
                    delta=f"${stats.get('total_pnl', 0):.2f}")
        col4.metric("Avg P&L", f"${stats.get('avg_pnl', 0):.2f}")
        
        coin_stats = stats.get('coin_stats', {})
        if coin_stats:
            st.caption("Per-Coin Performance")
            coin_df = pd.DataFrame([{
                'Coin': coin,
                'Wins': s['wins'],
                'Losses': s['losses'],
                'Win Rate': f"{s['win_rate']*100:.0f}%",
                'P&L': f"${s['pnl']:.2f}"
            } for coin, s in coin_stats.items()])
            st.dataframe(coin_df, use_container_width=True, hide_index=True)
else:
    st.info("📝 No settlement data yet. The model will track performance as you continue using it.")

st.divider()

# --- Next Contract Targets ---
st.markdown("### 🎯 Next Kalshi Contract Targets")

if TZ_AVAILABLE:
    now = datetime.now(pytz.timezone(LOCAL_TZ))
else:
    now = datetime.now()

next_settlement, minutes_until = get_next_kalshi_settlement(now)

st.caption(f"⏰ Next settlement at **{next_settlement.strftime('%I:%M %p CT')}** ({minutes_until}m remaining)")

target_cols = st.columns(len(COINS))

for idx, coin in enumerate(COINS):
    with target_cols[idx]:
        target = contract_targets.get(coin)
        if target:
            metadata = COIN_METADATA.get(coin, {'name': coin.replace('-USD', ''), 'symbol': coin.replace('-USD', ''), 'color': '#ffffff'})
            
            if target['direction'] == "UP":
                dir_class = "direction-up"
                dir_emoji = "⬆️"
            elif target['direction'] == "DOWN":
                dir_class = "direction-down"
                dir_emoji = "⬇️"
            else:
                dir_class = "direction-wait"
                dir_emoji = "⏳"
            
            conf_pct = target['win_prob'] * 100
            conf_color = '#00b894' if conf_pct >= 65 else '#fdcb6e' if conf_pct >= 55 else '#ff6b6b'
            
            st.markdown(f"""
            <div class="contract-card">
                <div style="font-size: 0.9rem; font-weight: 700; color: #ccc;">{metadata['name']} ({metadata['symbol']})</div>
                <div style="font-size: 0.7rem; color: #888;">Target: {next_settlement.strftime('%I:%M %p CT')}</div>
                <div class="direction {dir_class}" style="font-size: 1.2rem;">
                    {dir_emoji} {target['direction']}
                </div>
                <div class="target-price">
                    ${target['target_price']:.2f}
                </div>
                <div style="font-size: 0.8rem; color: #888;">
                    Move: <span style="color: {'#00b894' if target['projected_move_pct'] > 0 else '#ff6b6b'};">
                        {target['projected_move_pct']:+.1f}%
                    </span>
                </div>
                <div style="font-size: 0.8rem; color: #888;">
                    Confidence: {target['win_prob']:.0%}
                </div>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: {conf_pct:.0f}%; background: {conf_color};"></div>
                </div>
                <div style="font-size: 0.8rem; font-weight: 600; margin-top: 0.3rem; color: {'#00b894' if target['signal'] == 'BUY YES' else '#ff6b6b' if target['signal'] == 'BUY NO' else '#636e72'};">
                    {target['signal']}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="contract-card">
                <div style="font-size: 0.9rem; font-weight: 700; color: #ccc;">{COIN_METADATA.get(coin, {}).get('name', coin)}</div>
                <div style="font-size: 0.8rem; color: #888;">No data available</div>
            </div>
            """, unsafe_allow_html=True)

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

st.markdown("### 🌍 Macro Indicators")
macro_col1, macro_col2, macro_col3 = st.columns(3)
macro_col1.metric("SPY", f"${macro_data.get('spy_close', 0):.2f}", delta=f"{macro_data.get('spy_change', 0)*100:.2f}%")
macro_col2.metric("VIX", f"{macro_data.get('vix_close', 0):.2f}", delta=f"{macro_data.get('vix_change', 0)*100:.2f}%")
macro_col3.metric("DXY", f"{macro_data.get('dxy_close', 0):.2f}", delta="—")

st.divider()

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

st.markdown("### 📈 Detailed Edge Analysis")
if all_results:
    df_display = pd.DataFrame([{
        'Coin': r['Name'],
        'Symbol': r['Symbol'],
        'Price': r['Price_Str'],
        'Kalshi Price': r['Market_Price_Str'],
        'Spread': r['Spread_Str'],
        'Spread Quality': r['Spread_Quality'],
        'Imbalance': r['Imbalance_10_Str'],
        'Depth Slope': r['Depth_Slope_Str'],
        'Depth Slope Label': r['Depth_Slope_Label'],
        'Liquidity': r['Liquidity'],
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
    st.warning("No predictions available.")

st.divider()

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
    st.markdown(f"**Bet Size:** ${BET_SIZE:.2f}")
    st.markdown(f"**Time Zone:** {LOCAL_TZ}")
    
    st.divider()
    
    st.markdown("### 🆕 Code 5.3 Features")
    st.markdown("""
    ✅ **Previous Settlement Comparison** — Shows predicted vs. actual price  
    ✅ **Hypothetical P&L** — $10 bet simulation per coin  
    ✅ **Win/Loss Tracking** — ✅ WIN or ❌ LOSS for each prediction  
    ✅ **Accuracy Tracker** — Running win rate and P&L statistics  
    ✅ **Per-Coin Performance** — See which coins your model predicts best  
    ✅ **Local Time Zone** — All times in Central Time (CT)  
    ✅ **Kalshi Schedule Alignment** — :00, :15, :30, :45 settlements  
    """)
    
    st.divider()
    
    st.markdown("### 📊 How It Works")
    st.markdown("""
    1. **Model predicts** direction for each coin  
    2. **At the next Kalshi settlement**, compares prediction to actual price  
    3. **Records result** — win/loss and hypothetical P&L  
    4. **Tracks performance** — shows your model's real accuracy  
    """)
    
    st.divider()
    
    st.markdown("### 🔄 Manual Refresh")
    st.caption("Press 'R' or click the refresh button in your browser.")

# --- Footer ---
st.divider()
st.caption(f"⚡ Code 5.3 • Settlement Backtest • All times in CT • Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
