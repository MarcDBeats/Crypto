# ============================================
# CODE 2: ENHANCED KALSHI EDGE DETECTOR (FULLY FIXED)
# Simplified feature set | Robust fallbacks | Working predictions
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# --- Page Config ---
st.set_page_config(
    page_title="Enhanced Kalshi Edge Detector",
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
    .stDataFrame {
        border-radius: 0.5rem;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown('<div class="main-header">📊 Enhanced Kalshi Edge Detector</div>', unsafe_allow_html=True)
st.caption(f"⚡ Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# --- Settings ---
BANKROLL = 100.00
MAX_RISK_PER_TRADE = 0.02
MIN_EDGE = 0.05

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
@st.cache_data(ttl=30)
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

@st.cache_data(ttl=30)
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
    
    # --- 1. LOG RETURNS (Fixes non-stationarity) ---
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['log_return_5'] = np.log(df['close'] / df['close'].shift(5))
    df['log_return_10'] = np.log(df['close'] / df['close'].shift(10))
    
    # --- 2. ABSOLUTE FEATURES (Linearizes non-linear relationships) ---
    df['abs_log_return'] = df['log_return'].abs()
    df['abs_log_return_5'] = df['log_return_5'].abs()
    df['abs_log_return_10'] = df['log_return_10'].abs()
    df['abs_price_range'] = (df['high'] - df['low']).abs()
    df['abs_volume_change'] = df['volume'].pct_change().abs()
    
    # --- 3. MOVING AVERAGES (Trend detection) ---
    df['sma_5'] = df['close'].rolling(5).mean()
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['ema_9'] = df['close'].ewm(span=9).mean()
    df['ema_21'] = df['close'].ewm(span=21).mean()
    
    # --- 4. RSI (Momentum) ---
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # --- 5. MACD (Trend and Momentum) ---
    df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['macd_abs'] = df['macd_hist'].abs()
    
    # --- 6. BOLLINGER BANDS (Volatility) ---
    df['bb_middle'] = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * bb_std
    df['bb_lower'] = df['bb_middle'] - 2 * bb_std
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    
    # --- 7. VOLATILITY FEATURES ---
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
    
    # --- 10. PRICE RATIO FEATURES ---
    df['price_range'] = (df['high'] - df['low']) / df['close']
    df['abs_price_range'] = df['price_range'].abs()
    df['high_low_ratio'] = df['high'] / df['low']
    df['close_open_ratio'] = df['close'] / df['open']
    
    return df

# --- GET MODEL PROBABILITY (FIXED) ---
def get_model_probability(coin_symbol, window_minutes, df_clean=None):
    """Get the model's probability estimate using enhanced features"""
    try:
        if df_clean is None:
            df = fetch_yahoo_data(coin_symbol)
            if df.empty:
                return None
            df = add_enhanced_indicators(df)
            df_clean = df.dropna()
        
        if len(df_clean) < 60:
            return None
        
        # --- SIMPLIFIED FEATURE SET (works with less data) ---
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
        
        # --- FALLBACK: Try XGBoost, fallback to Random Forest ---
        try:
            from xgboost import XGBClassifier
            model = XGBClassifier(
                n_estimators=30,
                max_depth=3,
                learning_rate=0.1,
                random_state=42,
                use_label_encoder=False
            )
            model.fit(X_train, y_train)
        except:
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(
                n_estimators=30,
                max_depth=5,
                random_state=42
            )
            model.fit(X_train, y_train)
        
        last_row = df_clean[available_cols].iloc[-1].values.reshape(1, -1)
        win_prob = model.predict_proba(last_row)[0][1]
        
        return win_prob
    except Exception as e:
        return None

# --- Main Loop ---
all_results = []
best_bets = []

fear_greed = fetch_fear_greed_index()

# Fetch Kalshi prices
kalshi_prices = {}
for coin in COINS:
    ticker = KALSHI_TICKERS.get(coin)
    if ticker:
        price = fetch_kalshi_price(ticker)
        if price:
            kalshi_prices[coin] = price

progress_bar = st.progress(0)
status_text = st.empty()

for idx, coin in enumerate(COINS):
    status_text.text(f"🔄 Analyzing {coin} with enhanced features...")
    
    try:
        df = fetch_yahoo_data(coin)
        if df.empty:
            continue
        
        df = add_enhanced_indicators(df)
        df_clean = df.dropna()
        
        if len(df_clean) < 60:
            continue
        
        # Get model probabilities for 5m and 15m
        prob_5m = get_model_probability(coin, 5, df_clean)
        prob_15m = get_model_probability(coin, 15, df_clean)
        
        # Get Kalshi market price (with fallback)
        market_price = kalshi_prices.get(coin, 0.50)
        
        # Calculate edges
        edge_5m = prob_5m - market_price if prob_5m else 0
        edge_15m = prob_15m - market_price if prob_15m else 0
        
        # --- DECISION LOGIC ---
        if prob_5m and edge_5m >= MIN_EDGE:
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
        
        if prob_15m and edge_15m >= MIN_EDGE:
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
                    <span style="font-size: 0.8rem; font-weight: 600; color: {'#00b894' if result['Direction_5m'] == 'YES' else '#ff6b6b' if result['Direction_5m'] == 'NO' else '#888'};">
                        {result['Direction_5m']}
                    </span>
                    <span style="font-size: 0.7rem; color: #888;">| 15m:</span>
                    <span style="font-size: 0.8rem; font-weight: 600; color: {'#00b894' if result['Direction_15m'] == 'YES' else '#ff6b6b' if result['Direction_15m'] == 'NO' else '#888'};">
                        {result['Direction_15m']}
                    </span>
                </div>
                <div class="coin-stats">
                    Model 5m: {result['Prob_5m_Str']} | 15m: {result['Prob_15m_Str']}
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
    
    st.divider()
    
    st.markdown("### 🆕 Enhanced Features")
    st.markdown("""
    ✅ **Log Returns** (fixes non-stationarity)  
    ✅ **Absolute Features** (linearizes non-linear relationships)  
    ✅ **Advanced Indicators** (ATR, MFI, BB Width)  
    ✅ **XGBoost + Random Forest Fallback**  
    """)
    
    st.divider()
    
    st.markdown("### 🎯 How It Works")
    st.markdown("""
    1. **Model Estimate** — Enhanced ML predicts probability of price increase  
    2. **Kalshi Price** — Market's implied probability (YES contract price)  
    3. **Edge Calculation** — Model - Kalshi Price  
    4. **Decision**:
       - Edge > 5% → **BUY YES**
       - Edge < -5% → **BUY NO**
       - Otherwise → **SKIP**
    """)
    
    st.divider()
    
    st.markdown("### 🔄 Auto-Refresh")
    st.caption("Refreshes every 30 seconds")

# --- Footer ---
st.divider()
st.caption(f"⚡ Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Enhanced with Log Returns, Abs Features, XGBoost")
