import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime

st.set_page_config(page_title="Crypto Prices", layout="wide")
st.title("🔮 Live Crypto Prices")
st.caption(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

@st.cache_data(ttl=60)
def get_price(symbol):
    ticker = yf.Ticker(symbol)
    data = ticker.history(period="1d", interval="1m")
    if data.empty:
        return None
    last = data.iloc[-1]
    return {
        'price': last['Close'],
        'volume': last['Volume']
    }

coins = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD', 'DOGE-USD']

results = []
for coin in coins:
    info = get_price(coin)
    if info:
        results.append({
            'Coin': coin.replace('-USD', ''),
            'Price': f"${info['price']:.2f}",
            'Volume': f"{info['volume']:,.0f}"
        })

if results:
    df = pd.DataFrame(results)
    st.dataframe(df, use_container_width=True, hide_index=True)
else:
    st.warning("No data available")

st.caption("Auto-refreshes every 60 seconds")
