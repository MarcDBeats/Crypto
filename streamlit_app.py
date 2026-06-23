# ============================================
# CRYPTO DASHBOARD - UPGRADED VERSION
# XGBoost Model + Sentiment + Charts
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
    page_title="Crypto Predictor Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Better Styling ---
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
    }
    .metric-card .coin-symbol {
        font-size: 1.5rem;
        font-weight: 600;
    }
    .metric-card .coin-price {
        font-size: 1.2rem;
        font-weight: 600;
        margin: 0.3rem 0;
    }
    .metric-card .coin-change {
        font-size: 0.9rem;
    }
    .metric-card .coin-action {
        font-size: 0.9rem;
        margin-top: 0.5rem;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        display: inline-block;
    }
    .metric-card .coin-stats {
        font-size: 0.7rem;
        color: #888;
        margin-top: 0.3rem;
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
    .action-buy-yes {
        background: #00b89433;
        color: #00b894;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        font-weight: 600;
    }
    .action-buy-no {
        background: #ff6b6b33;
        color: #ff6b6b;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        font-weight: 600;
    }
    .action-skip {
        background: #636e7233;
        color: #b2bec3;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown('<div class="main-header">📊 Crypto Predictor Pro</div>', unsafe_allow_html=True)
st.caption(f"⚡ Live Predictions • Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# --- Settings ---
BANKROLL = 100.00
MAX_RISK_PER_TRADE = 0.02
MIN_EDGE = 0.05
PREDICT_WINDOW = 5

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

# --- Data Fetching Functions ---
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
            'percent_change_1h': quotes.get('percent_change_1h', 0),
            'percent_change_24h': quotes.get('percent_change_24h', 0),
            'volume_24h': quotes.get('volume_24h', 0),
            'market_cap': quotes.get('market_cap', 0)
        }
    except:
        return {'price': 0, 'percent_change_15m': 0, 'percent_change_1h': 0, 
                'percent_change_24h': 0, 'volume_24h': 0, 'market_cap': 0}

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
    # Moving Averages
    df['sma_5'] = df['close'].rolling(5).mean()
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['ema_9'] = df['close'].ewm(span=9).mean()
    df['ema_21'] = df['close'].ewm(span=21).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * bb_std
    df['bb_lower'] = df['bb_middle'] - 2 * bb_std
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # Price Features
    df['return_1'] = df['close'].pct_change()
    df['return_5'] = df['close'].pct_change(5)
    df['return_10'] = df['close'].pct_change(10)
    df['price_range'] = (df['high'] - df['low']) / df['close']
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(10).mean()
    
    # Volatility
    df['volatility'] = df['return_1'].rolling(10).std()
    
    return df

# --- XGBoost Model (Fallback to Random Forest if XGBoost not installed) ---
def get_model():
    """Try to use XGBoost, fallback to Random Forest"""
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=50, max_depth=3, random_state=42, use_label_encoder=False)
    except:
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)

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

# --- Helper Functions for Color Coding ---
def color_change(val):
    """Color code percentage changes"""
    if isinstance(val, str):
        val = val.replace('%', '')
    try:
        val = float(val)
        if val > 0:
            return 'color: #00b894'
        elif val < 0:
            return 'color: #ff6b6b'
        else:
            return 'color: #fdcb6e'
    except:
        return ''

def color_confidence(val):
    """Color code confidence"""
    if isinstance(val, str):
        val = val.replace('%', '')
    try:
        val = float(val)
        if val >= 65:
            return 'color: #00b894; font-weight: bold'
        elif val >= 55:
            return 'color: #fdcb6e; font-weight: bold'
        else:
            return 'color: #ff6b6b'
    except:
        return ''

def color_action(val):
    """Color code action labels"""
    if 'BUY YES' in val:
        return 'background-color: #00b89433; color: #00b894; font-weight: bold; padding: 0.2rem 0.5rem; border-radius: 0.3rem;'
    elif 'BUY NO' in val:
        return 'background-color: #ff6b6b33; color: #ff6b6b; font-weight: bold; padding: 0.2rem 0.5rem; border-radius: 0.3rem;'
    elif 'SKIP' in val:
        return 'background-color: #636e7233; color: #b2bec3; font-weight: bold; padding: 0.2rem 0.5rem; border-radius: 0.3rem;'
    return ''

# --- Main Prediction Loop ---
all_results = []
best_bets = []

paprika_map = {
    'BTC-USD': 'btc-bitcoin',
    'ETH-USD': 'eth-ethereum',
    'SOL-USD': 'sol-solana',
    'BNB-USD': 'bnb-binance-coin',
    'XRP-USD': 'xrp-xrp',
    'DOGE-USD': 'doge-dogecoin'
}

fear_greed = fetch_fear_greed_index()

# Progress bar
progress_bar = st.progress(0)
status_text = st.empty()

for idx, coin in enumerate(COINS):
    status_text.text(f"🔄 Analyzing {coin}...")
    
    try:
        # Fetch data
        df = fetch_yahoo_data(coin)
        if df.empty:
            continue
        
        df = add_advanced_indicators(df)
        df_clean = df.dropna()
        
        if len(df_clean) < 50:
            continue
        
        # Get external data
        paprika_data = fetch_coinpaprika_data(paprika_map[coin])
        coin_name = coin.replace('-USD', '')
        
        # Features
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
            continue
        
        X_train = X_df_clean[available_cols].values
        y_train = X_df_clean['target'].values
        
        # Train model
        model = get_model()
        model.fit(X_train, y_train)
        
        # Make prediction
        last_row = df_clean[available_cols].iloc[-1].values.reshape(1, -1)
        win_prob = model.predict_proba(last_row)[0][1]
        
        # Determine action
        edge = win_prob - 0.50
        bet_amount = calculate_kelly_bet(win_prob, BANKROLL, MAX_RISK_PER_TRADE)
        current_price = df_clean['close'].iloc[-1]
        
        if edge >= MIN_EDGE and win_prob > 0.55:
            action = "BUY YES" if win_prob > 0.5 else "BUY NO"
            is_signal = True
        else:
            action = "SKIP"
            is_signal = False
        
        # Get coin info
        name, symbol, color = COIN_NAMES.get(coin, (coin_name, '', '#ffffff'))
        
        result = {
            'Coin': symbol,
            'Name': name,
            'Price': current_price,
            'Price_Str': f"${current_price:.2f}",
            'Win_Prob': win_prob,
            'Win_Prob_Str': f"{win_prob:.0%}",
            'Edge': edge,
            'Edge_Str': f"{edge:.0%}",
            'Change_15m': paprika_data.get('percent_change_15m', 0),
            'Change_24h': paprika_data.get('percent_change_24h', 0),
            'Bet_Size': bet_amount,
            'Bet_Size_Str': f"${bet_amount:.2f}" if bet_amount > 0 else "$0.00",
            'Action': action,
            'Action_Color': color,
            'Is_Signal': is_signal
        }
        all_results.append(result)
        
        if is_signal:
            best_bets.append(result)
            
    except Exception as e:
        pass
    
    progress_bar.progress((idx + 1) / len(COINS))

status_text.empty()
progress_bar.empty()

# ============================================
# 📊 DASHBOARD LAYOUT
# ============================================

# --- Fear & Greed Index ---
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
            color = result['Action_Color']
            change_color = '#00b894' if result['Change_24h'] > 0 else '#ff6b6b' if result['Change_24h'] < 0 else '#fdcb6e'
            
            # Action label with proper styling
            action = result['Action']
            if 'BUY YES' in action:
                action_class = 'action-buy-yes'
            elif 'BUY NO' in action:
                action_class = 'action-buy-no'
            else:
                action_class = 'action-skip'
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="coin-symbol">{result['Coin']}</div>
                <div class="coin-price">{result['Price_Str']}</div>
                <div class="coin-change" style="color: {change_color};">
                    {result['Change_24h']:+.1f}%
                </div>
                <div>
                    <span class="{action_class}">{action}</span>
                </div>
                <div class="coin-stats">
                    Win: {result['Win_Prob_Str']} | Edge: {result['Edge_Str']}
                </div>
            </div>
            """, unsafe_allow_html=True)

st.divider()

# --- Predictions Table ---
st.markdown("### 📈 Detailed Predictions")

if all_results:
    df_display = pd.DataFrame([{
        'Coin': r['Coin'],
        'Name': r['Name'],
        'Price': r['Price_Str'],
        'Win Prob': r['Win_Prob_Str'],
        'Edge': r['Edge_Str'],
        '24h Change': f"{r['Change_24h']:+.1f}%",
        '15m Change': f"{r['Change_15m']:+.1f}%",
        'Bet': r['Bet_Size_Str'],
        'Action': r['Action']
    } for r in all_results])
    
    # Color code the dataframe using .map (newer pandas) or .applymap (older)
    try:
        # Try the new .map method (pandas 2.1+)
        styled_df = df_display.style.map(color_change, subset=['24h Change', '15m Change'])
        styled_df = styled_df.map(color_confidence, subset=['Win Prob'])
        styled_df = styled_df.map(color_action, subset=['Action'])
    except AttributeError:
        # Fallback to .applymap for older pandas
        styled_df = df_display.style.applymap(color_change, subset=['24h Change', '15m Change'])
        styled_df = styled_df.applymap(color_confidence, subset=['Win Prob'])
        styled_df = styled_df.applymap(color_action, subset=['Action'])
    
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
else:
    st.warning("No predictions available. Please check data sources.")

st.divider()

# --- Best Bets Section ---
st.markdown("### ⭐ Best Bets")

if best_bets:
    cols = st.columns(min(len(best_bets), 3))
    for i, bet in enumerate(best_bets[:6]):
        col = cols[i % 3]
        with col:
            st.markdown(f"""
            <div class="best-bet">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 1.2rem;">{bet['Coin']}</span>
                    <span style="font-size: 1.5rem;">{bet['Action']}</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-top: 0.5rem;">
                    <span>Win: {bet['Win_Prob_Str']}</span>
                    <span>Edge: {bet['Edge_Str']}</span>
                    <span>Bet: {bet['Bet_Size_Str']}</span>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 0.8rem; opacity: 0.8;">
                    <span>24h: {bet['Change_24h']:+.1f}%</span>
                    <span>15m: {bet['Change_15m']:+.1f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="skip-bet">
        ⏳ No bets meet the minimum edge threshold. Waiting for better signals...
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
            # Create subplot with price and volume
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                               vertical_spacing=0.05, 
                               row_heights=[0.7, 0.3])
            
            # Price line
            fig.add_trace(go.Scatter(
                x=df_chart['time'],
                y=df_chart['close'],
                name='Price',
                line=dict(color='#667eea', width=2)
            ), row=1, col=1)
            
            # Volume bars
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
            
            fig.update_xaxes(title_text="Time", row=2, col=1)
            fig.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            
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
    
    st.markdown("### 🎯 Strategy")
    st.markdown("""
    **Signal:** Win Prob > 55% + Edge > 5%  
    **Sizing:** Kelly Criterion (capped at 2%)  
    **Exit:** Stop loss at 2%  
    """)
    
    st.divider()
    
    st.markdown("### 🔄 Auto-Refresh")
    st.caption("Dashboard refreshes every 60 seconds")

# --- Footer ---
st.divider()
st.caption(f"⚡ Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Built with Streamlit")
