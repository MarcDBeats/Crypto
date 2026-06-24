@st.cache_data(ttl=10)
def fetch_kalshi_market_data(ticker):
    """Fetch full market data including strike price and probabilities"""
    try:
        url = f"https://external-api.kalshi.com/trade-api/v2/markets/{ticker}"
        response = requests.get(url, timeout=5)
        data = response.json()
        market = data.get('market', {})
        
        # Get the strike price and contract details
        strike_price = market.get('strike_price', 0)
        yes_bid = market.get('yes_bid', 0)
        yes_ask = market.get('yes_ask', 0)
        implied_prob = (yes_bid + yes_ask) / 2 if yes_bid > 0 and yes_ask > 0 else 0.50
        
        return {
            'strike_price': strike_price,
            'implied_probability': implied_prob,
            'yes_bid': yes_bid,
            'yes_ask': yes_ask
        }
    except:
        return None
