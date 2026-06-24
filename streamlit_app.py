# ============================================
# CODE 6: CLOSE-TO-SETTLEMENT PRICE PREDICTOR
# Only trades within 5 minutes of Kalshi settlement
# Shows model price prediction for each 15-min interval
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
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown('<div class="main-header">📊 Kalshi Price Predictor</div>', unsafe_allow_html=True)
st.caption(f"⚡ Only trades within 5 minutes of settlement • Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# --- Settings ---
MIN_EDGE = 0.05
FLAT_THRESHOLD = 0.002
PREDICT_WINDOW = 5  # Only trade within 5 minutes of settlement
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
def get_next_kalshi_settlement():
    """Get the next :00, :15, :30, :45 mark"""
    now = datetime.now()
    minute = now.minute
    minutes_to_next = (15 - (minute % 15)) % 15
    if minutes_to_next == 0:
        minutes_to_next = 15
    next_settlement = now + timedelta(minutes=minutes_to_next)
    next_settlement = next_settlement.replace(second=0, microsecond=0)
    return next_settlement, minutes_to_next

def get_previous_kalshi_settlement():
    """Get the previous :00, :15, :30, :45 mark"""
    now = datetime.now()
    minute = now.minute
    minutes_since = minute % 15
    if minutes_since == 0:
        minutes_since = 15
    prev_settlement = now - timedelta(minutes=minutes_since)
    prev_settlement = prev_settlement.replace(second=0, microsecond=0)
    return prev_settlement

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

# --- Feature Engineering ---
def add_advanced_features(df):
    """Add technical indicators"""
    
    df['return_1'] = df['close'].pct_change()
    df['return_5'] = df['close'].pct_change(5)
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['abs_log_return'] = df['log_return'].abs()
    df['lag_1'] = df['close'].shift(1)
    df['lag_5'] = df['close'].shift(5)
    df['volatility_5'] = df['return_1'].rolling(5).std()
    df['volatility_10'] = df['return_1'].rolling(10).std()
    df['sma_5'] = df['close'].rolling(5).mean()
    df['sma_10'] = df['close'].rolling(10).mean()
    df['rsi'] = 100 - (100 / (1 + (df['close'].diff().clip(lower=0).rolling(14).mean() / (-df['close'].diff().clip(upper=0).rolling(14).mean() + 0.001))))
    df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    df['macd_hist'] = df['macd'] - df['macd'].ewm(span=9).mean()
    df['bb_position'] = (df['close'] - df['close'].rolling(20).mean() + 2 * df['close'].rolling(20).std()) / (4 * df['close'].rolling(20).std())
    df['atr'] = (df['high'] - df['low']).rolling(14).mean()
    df['price_range'] = (df['high'] - df['low']) / df['close']
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(10).mean()
    
    return df

# --- Get Model Prediction for Next Settlement ---
def get_settlement_prediction(coin_symbol):
    try:
        df = fetch_yahoo_data(coin_symbol)
        if df.empty:
            return None
        
        df = add_advanced_features(df)
        df_clean = df.dropna()
        
        if len(df_clean) < 60:
            return None
        
        # Get current price and next settlement time
        current_price = df_clean['close'].iloc[-1]
        
        # Get minutes to next settlement
        _, minutes_until = get_next_kalshi_settlement()
        
        # If more than 10 minutes away, use 15-minute window
        if minutes_until > PREDICT_WINDOW + 5:
            minutes_until = 15
        
        # Features
        feature_cols = [
            'close', 'volume', 'log_return', 'abs_log_return',
            'lag_1', 'lag_5', 'volatility_5', 'volatility_10',
            'sma_5', 'sma_10', 'rsi', 'macd_hist',
            'bb_position', 'atr', 'price_range', 'volume_ratio'
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
        
        # Calculate target price
        if win_prob > 0.5:
            target_price = current_price * (1 + (win_prob - 0.5) * 0.02)
        else:
            target_price = current_price * (1 - (0.5 - win_prob) * 0.02)
        
        # Get Kalshi price
        ticker = KALSHI_TICKERS.get(coin_symbol, '')
        kalshi_price = fetch_kalshi_price(ticker) if ticker else 0.50
        
        # Calculate edge against Kalshi
        edge = win_prob - kalshi_price
        
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
            'kalshi_price': kalshi_price
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

# --- Only show predictions if within trading window ---
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

# --- Explanation ---
st.markdown("### 📖 How This Works")
st.markdown("""
**Trading Window**: Only trades are shown when you're within **5 minutes** of the next Kalshi settlement.

**Price Predictions**:
- **Target Price**: What the model expects the price to be at settlement
- **Current Price**: The current spot price
- **Kalshi Price**: Kalshi's implied probability (contract price)

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
st.caption(f"⚡ Code 6 • Close-to-Settlement Predictor • Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
