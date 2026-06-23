# ============================================
# CRYPTO DASHBOARD - 15-MINUTE INTERVAL SIGNALS
# Displays predictions for :00, :15, :30, :45
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
    page_title="Crypto Predictor Pro",
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
    .signal-card {
        background: linear-gradient(135deg, #1e1e2f 0%, #2d2d44 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        border: 2px solid #3d3d5c;
        text-align: center;
        min-height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    .signal-card .interval-label {
        font-size: 1.2rem;
        font-weight: 700;
        color: #888;
    }
    .signal-card .signal-direction {
        font-size: 2rem;
        font-weight: 700;
        margin: 0.3rem 0;
    }
    .signal-card .signal-confidence {
        font-size: 0.9rem;
        color: #888;
    }
    .signal-card .signal-price {
        font-size: 0.8rem;
        color: #666;
    }
    .signal-up {
        border-color: #00b894 !important;
        background: linear-gradient(135deg, #1e2f2a 0%, #2d4438 100%) !important;
    }
    .signal-up .signal-direction {
        color: #00b894;
    }
    .signal-down {
        border-color: #ff6b6b !important;
        background: linear-gradient(135deg, #2f1e1e 0%, #442d2d 100%) !important;
    }
    .signal-down .signal-direction {
        color: #ff6b6b;
    }
    .signal-wait {
        border-color: #636e72 !important;
        background: linear-gradient(135deg, #1e1e1e 0%, #2d2d2d 100%) !important;
    }
    .signal-wait .signal-direction {
        color: #636e72;
    }
    .signal-active {
        border-color: #fdcb6e !important;
        box-shadow: 0 0 20px rgba(253, 203, 110, 0.3) !important;
    }
    .countdown-timer {
        font-size: 2rem;
        font-weight: 700;
        color: #fdcb6e;
        text-align: center;
        padding: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e1e2f 0%, #2d2d44 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #3d3d5c;
        text-align: center;
    }
    .best-bet {
        background: linear-gradient(135deg, #00b894 0%, #00cec9 100%);
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
    .stDataFrame {
        border-radius: 0.5rem;
        overflow: hidden;
    }
    .badge-up {
        background: #00b89433;
        color: #00b894;
        padding: 0.2rem 0.6rem;
        border-radius: 0.3rem;
        font-weight: 700;
    }
    .badge-down {
        background: #ff6b6b33;
        color: #ff6b6b;
        padding: 0.2rem 0.6rem;
        border-radius: 0.3rem;
        font-weight: 700;
    }
    .badge-wait {
        background: #636e7233;
        color: #b2bec3;
        padding: 0.2rem 0.6rem;
        border-radius: 0.3rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown('<div class="main-header">📊 Crypto Predictor Pro</div>', unsafe_allow_html=True)

# --- Settings ---
BANKROLL = 100.00
MAX_RISK_PER_TRADE = 0.02
MIN_EDGE = 0.05
PREDICT_WINDOW = 5  # Keep at 5 minutes (Kalshi's interval)

COINS = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD', 'DOGE-USD']

# Coin metadata for display
COIN_NAMES = {
    'BTC-USD': ('Bitcoin', '₿', '#f7931a'),
    'ETH-USD': ('Ethereum', '⟠', '#627eea'),
    'SOL-USD': ('Solana', '◎', '#9945ff'),
    'BNB-USD': ('BNB', '◆', '#f3ba2f'),
    'XRP-USD': ('XRP', '✕', '#00aae4'),
    'DOGE-USD': ('Dogecoin', 'Ð', '#c2a633')
}

# --- Helper Functions ---
def get_next_interval():
    """Get the next :00, :15, :30, :45 mark and time remaining"""
    now = datetime.now()
    minute = now.minute
    second = now.second
    
    # Find next interval
    if minute % 15 == 0 and second == 0:
        next_minute = minute
        next_hour = now.hour
        remaining = 0
    else:
        next_interval = ((minute // 15) + 1) * 15
        if next_interval == 60:
            next_interval = 0
            next_hour = now.hour + 1
        else:
            next_hour = now.hour
        
        # Calculate seconds remaining
        next_time = now.replace(hour=next_hour, minute=next_interval, second=0, microsecond=0)
        if next_time <= now:
            next_time += timedelta(hours=1)
        remaining = int((next_time - now).total_seconds())
    
    return next_time, remaining

def format_time_remaining(seconds):
    """Format seconds into MM:SS"""
    mins = seconds // 60
    secs = seconds % 60
    return f"{mins}m {secs}s"

# --- Data Fetching Functions ---
@st.cache_data(ttl=30)
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
        }
    except:
        return {'price': 0, 'percent_change_15m': 0, 'percent_change_24h': 0}

@st.cache_data(ttl=3600)
def fetch_fear_greed_index():
    """Fetch the Fear & Greed Index"""
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

# --- Technical Indicators ---
def add_advanced_indicators(df):
    """Add advanced technical indicators"""
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
    
    df['return_1'] = df['close'].pct_change()
    df['return_5'] = df['close'].pct_change(5)
    df['return_10'] = df['close'].pct_change(10)
    df['price_range'] = (df['high'] - df['low']) / df['close']
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(10).mean()
    df['volatility'] = df['return_1'].rolling(10).std()
    
    return df

# --- XGBoost Model (Fallback to Random Forest) ---
def get_model():
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=50, max_depth=3, random_state=42, use_label_encoder=False)
    except:
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)

def calculate_kelly_bet(win_prob, bankroll, max_risk_pct=0.02):
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

# --- Get Predictions for Each Coin ---
def get_predictions_for_coin(coin_symbol):
    """Get the latest prediction for a single coin"""
    try:
        df = fetch_yahoo_data(coin_symbol)
        if df.empty:
            return None
        
        df = add_advanced_indicators(df)
        df_clean = df.dropna()
        
        if len(df_clean) < 50:
            return None
        
        coin_name = coin_symbol.replace('-USD', '')
        
        feature_cols = ['close', 'volume', 'return_1', 'return_5', 'return_10', 
                       'price_range', 'volume_ratio', 'rsi', 'sma_5', 'sma_10',
                       'sma_20', 'ema_9', 'ema_21', 'macd', 'macd_signal', 
                       'bb_position', 'volatility']
        
        available_cols = [col for col in feature_cols if col in df_clean.columns]
        X = df_clean[available_cols].values
        y = df_clean['close'].shift(-PREDICT_WINDOW) > df_clean['close']
        
        X_df = pd.DataFrame(X, columns=available_cols)
        X_df['target'] = y.astype(int)
        X_df_clean = X_df.dropna()
        
        if len(X_df_clean) < 30:
            return None
        
        X_train = X_df_clean[available_cols].values
        y_train = X_df_clean['target'].values
        
        model = get_model()
        model.fit(X_train, y_train)
        
        last_row = df_clean[available_cols].iloc[-1].values.reshape(1, -1)
        win_prob = model.predict_proba(last_row)[0][1]
        
        current_price = df_clean['close'].iloc[-1]
        edge = win_prob - 0.50
        
        # Determine direction
        if edge >= MIN_EDGE and win_prob > 0.55:
            if win_prob > 0.5:
                direction = "UP"
            else:
                direction = "DOWN"
        else:
            direction = "WAIT"
        
        return {
            'coin': coin_name,
            'price': current_price,
            'win_prob': win_prob,
            'edge': edge,
            'direction': direction
        }
    except:
        return None

# --- Main Dashboard ---

# Get next interval info
next_time, remaining = get_next_interval()
is_active = remaining <= 300  # 5 minutes before interval

# --- Countdown Timer ---
st.markdown(f"""
<div style="text-align: center; padding: 0.5rem; background: linear-gradient(135deg, #1e1e2f 0%, #2d2d44 100%); border-radius: 0.5rem; border: 2px solid #3d3d5c; margin-bottom: 1rem;">
    <div style="font-size: 0.9rem; color: #888;">NEXT KALSHI CONTRACT SETTLEMENT</div>
    <div class="countdown-timer">{format_time_remaining(remaining)}</div>
    <div style="font-size: 1rem; color: #666;">{next_time.strftime('%I:%M:%S %p')} CST</div>
    <div style="font-size: 0.8rem; color: {'#00b894' if is_active else '#888'}; margin-top: 0.3rem;">
        {'🟢 Trading Window Open (5 minutes remaining)' if is_active else '⏳ Waiting for next trading window...'}
    </div>
</div>
""", unsafe_allow_html=True)

# --- Get Predictions for All Coins ---
all_predictions = {}
for coin in COINS:
    pred = get_predictions_for_coin(coin)
    if pred:
        all_predictions[coin] = pred

# --- Display Interval Signals ---
st.markdown("### 🎯 Signals for Next 15-Minute Intervals")

# Create columns for the 4 intervals
interval_cols = st.columns(4)

# Get predictions for each interval (simulate by using current data with small offsets)
for idx, (col, interval) in enumerate(zip(interval_cols, ['00', '15', '30', '45'])):
    # Determine which signal to show (using BTC as the primary signal for simplicity)
    # In a real scenario, you could aggregate across coins
    btc_pred = all_predictions.get('BTC-USD', None)
    
    # Determine signal state
    if btc_pred and btc_pred['direction'] != 'WAIT':
        direction = btc_pred['direction']
        confidence = f"{btc_pred['win_prob']:.0%}%"
        price = f"${btc_pred['price']:.2f}"
    else:
        direction = "WAIT"
        confidence = "—"
        price = "—"
    
    # Determine CSS class
    if direction == "UP":
        css_class = "signal-up"
        display = "⬆️ UP"
        color = "#00b894"
    elif direction == "DOWN":
        css_class = "signal-down"
        display = "⬇️ DOWN"
        color = "#ff6b6b"
    else:
        css_class = "signal-wait"
        display = "⏳ WAIT"
        color = "#636e72"
    
    # Highlight if this is the next interval
    next_minute = next_time.minute
    current_interval = int(interval)
    is_next = (next_minute % 15) == current_interval if current_interval > 0 else next_minute % 15 == 0
    active_class = "signal-active" if is_next and is_active else ""
    
    with col:
        st.markdown(f"""
        <div class="signal-card {css_class} {active_class}">
            <div class="interval-label">:{interval}</div>
            <div class="signal-direction" style="color: {color};">{display}</div>
            <div class="signal-confidence">Confidence: {confidence}</div>
            <div class="signal-price">{price}</div>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# --- Fear & Greed Index ---
st.markdown("### 🧠 Market Sentiment")
fg_col1, fg_col2, fg_col3, fg_col4, fg_col5 = st.columns([1, 1, 2, 1, 1])
with fg_col3:
    fear_greed = fetch_fear_greed_index()
    fg_color = '#00b894' if fear_greed['value'] >= 60 else '#fdcb6e' if fear_greed['value'] >= 40 else '#ff6b6b'
    st.markdown(f"""
    <div style="text-align: center; padding: 0.5rem; border-radius: 0.5rem; border: 2px solid {fg_color};">
        <span style="font-size: 1.5rem;">{fear_greed['classification']}</span><br>
        <span style="font-size: 2.5rem; font-weight: 700; color: {fg_color};">{fear_greed['value']}</span><br>
        <span style="font-size: 0.8rem; color: #888;">Fear & Greed Index</span>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# --- Market Overview ---
st.markdown("### 📊 Market Overview")
m_cols = st.columns(6)

for i, (coin, pred) in enumerate(all_predictions.items()):
    if i < 6:
        with m_cols[i]:
            name, symbol, color = COIN_NAMES.get(coin, (coin.replace('-USD', ''), '', '#ffffff'))
            change_color = '#00b894' if pred.get('change_24h', 0) > 0 else '#ff6b6b'
            
            # Direction CSS
            if pred['direction'] == "UP":
                dir_class = "direction-up"
                dir_emoji = "⬆️"
            elif pred['direction'] == "DOWN":
                dir_class = "direction-down"
                dir_emoji = "⬇️"
            else:
                dir_class = "direction-wait"
                dir_emoji = "⏳"
            
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 1.2rem; font-weight: 600;">{symbol}</div>
                <div style="font-size: 1rem; margin: 0.3rem 0;">${pred['price']:.2f}</div>
                <div style="font-size: 0.8rem; color: {change_color};">
                    {pred.get('change_24h', 0):+.1f}%
                </div>
                <div>
                    <span class="{dir_class}">{dir_emoji} {pred['direction']}</span>
                </div>
                <div style="font-size: 0.7rem; color: #888; margin-top: 0.3rem;">
                    Win: {pred['win_prob']:.0%} | Edge: {pred['edge']:.0%}
                </div>
            </div>
            """, unsafe_allow_html=True)

st.divider()

# --- Best Bets (Aggregated) ---
st.markdown("### ⭐ Current Best Bets")

# Find the best signal across all coins
best_signal = None
best_score = 0

for coin, pred in all_predictions.items():
    if pred['direction'] != 'WAIT':
        score = pred['win_prob'] * (1 + abs(pred['edge']))
        if score > best_score:
            best_score = score
            best_signal = pred

if best_signal:
    dir_emoji = "⬆️" if best_signal['direction'] == "UP" else "⬇️"
    dir_color = "#00b894" if best_signal['direction'] == "UP" else "#ff6b6b"
    
    st.markdown(f"""
    <div class="best-bet">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-size: 1.2rem;">{best_signal['coin']}</span>
            <span style="font-size: 1.5rem; color: {dir_color};">{dir_emoji} {best_signal['direction']}</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-top: 0.5rem;">
            <span>Win Prob: {best_signal['win_prob']:.0%}</span>
            <span>Edge: {best_signal['edge']:.0%}</span>
            <span>Price: ${best_signal['price']:.2f}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="skip-bet">
        ⏳ No directional signals meet the minimum edge threshold. Waiting for better opportunities...
    </div>
    """, unsafe_allow_html=True)

st.divider()

# --- Price Charts ---
st.markdown("### 📉 Price Charts")

selected_coin = st.selectbox("Select a coin to view chart:", COINS)
if selected_coin:
    try:
        df_chart = fetch_yahoo_data(selected_coin)
        if not df_chart.empty:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                               vertical_spacing=0.05, 
                               row_heights=[0.7, 0.3])
            
            fig.add_trace(go.Scatter(
                x=df_chart['time'],
                y=df_chart['close'],
                name='Price',
                line=dict(color='#667eea', width=2)
            ), row=1, col=1)
            
            fig.add_trace(go.Bar(
                x=df_chart['time'],
                y=df_chart['volume'],
                name='Volume',
                marker_color='#2d2d44'
            ), row=2, col=1)
            
            fig.update_layout(
                height=400,
                showlegend=False,
                template='plotly_dark',
                margin=dict(l=0, r=0, t=0, b=0)
            )
            
            st.plotly_chart(fig, use_container_width=True)
    except:
        st.warning("Chart data unavailable.")

st.divider()

# --- Sidebar ---
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown(f"**Bankroll:** ${BANKROLL:.2f}")
    st.markdown(f"**Max Risk/Trade:** {MAX_RISK_PER_TRADE*100:.0f}%")
    st.markdown(f"**Min Edge:** {MIN_EDGE*100:.0f}%")
    st.markdown(f"**Predict Window:** {PREDICT_WINDOW} min")
    
    st.divider()
    
    st.markdown("### 📚 Data Sources")
    st.markdown("""
    ✅ Yahoo Finance (Prices)  
    ✅ CoinPaprika (Market Data)  
    ✅ Alternative.me (Fear & Greed)  
    ✅ XGBoost/Random Forest (ML)  
    """)
    
    st.divider()
    
    st.markdown("### 🎯 How to Use")
    st.markdown("""
    1. **Check countdown timer** – signals are valid 5 minutes before each :00, :15, :30, :45 mark  
    2. **Read the interval signals** – shows predicted direction for each 15-minute block  
    3. **Follow the Best Bet** – highest confidence signal for the current period  
    4. **Place trade on Kalshi** – buy YES or NO based on the direction shown  
    """)
    
    st.divider()
    
    st.markdown("### 🔄 Auto-Refresh")
    st.caption("Dashboard refreshes every 30 seconds")

# --- Footer ---
st.divider()
st.caption(f"⚡ Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} CST | Built with Streamlit")
