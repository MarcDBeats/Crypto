{\rtf1\ansi\ansicpg1252\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fnil\fcharset0 Menlo-Regular;}
{\colortbl;\red255\green255\blue255;\red91\green100\blue110;\red255\green255\blue255;\red24\green24\blue24;
\red157\green0\blue210;\red10\green58\blue158;\red11\green34\blue86;\red129\green40\blue2;\red0\green0\blue0;
\red19\green118\blue70;\red0\green0\blue255;\red109\green51\blue215;\red194\green11\blue35;\red32\green108\blue135;
\red101\green76\blue29;}
{\*\expandedcolortbl;;\cssrgb\c43137\c46667\c50588;\cssrgb\c100000\c100000\c100000;\cssrgb\c12549\c12549\c12549;
\cssrgb\c68627\c0\c85882;\cssrgb\c1961\c31373\c68235;\cssrgb\c3922\c18824\c41176;\cssrgb\c58431\c21961\c0;\cssrgb\c0\c0\c0;
\cssrgb\c3529\c52549\c34510;\cssrgb\c0\c0\c100000;\cssrgb\c50980\c31373\c87451;\cssrgb\c81176\c13333\c18039;\cssrgb\c14902\c49804\c60000;
\cssrgb\c47451\c36863\c14902;}
\margl1440\margr1440\vieww29660\viewh16880\viewkind0
\deftab720
\pard\pardeftab720\partightenfactor0

\f0\fs24 \cf2 \cb3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 # ============================================\cf4 \cb1 \strokec4 \
\cf2 \cb3 \strokec2 # CODE 1 DASHBOARD - STREAMLIT VERSION\cf4 \cb1 \strokec4 \
\cf2 \cb3 \strokec2 # Optimized GRU Model with Yahoo Finance\cf4 \cb1 \strokec4 \
\cf2 \cb3 \strokec2 # ============================================\cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf5 \cb3 \strokec5 import\cf4 \strokec4  streamlit \cf5 \strokec5 as\cf4 \strokec4  st\cb1 \
\cf5 \cb3 \strokec5 import\cf4 \strokec4  pandas \cf5 \strokec5 as\cf4 \strokec4  pd\cb1 \
\cf5 \cb3 \strokec5 import\cf4 \strokec4  numpy \cf5 \strokec5 as\cf4 \strokec4  np\cb1 \
\cf5 \cb3 \strokec5 from\cf4 \strokec4  sklearn.preprocessing \cf5 \strokec5 import\cf4 \strokec4  MinMaxScaler\cb1 \
\cf5 \cb3 \strokec5 import\cf4 \strokec4  tensorflow \cf5 \strokec5 as\cf4 \strokec4  tf\cb1 \
\cf5 \cb3 \strokec5 from\cf4 \strokec4  tensorflow.keras.models \cf5 \strokec5 import\cf4 \strokec4  Sequential\cb1 \
\cf5 \cb3 \strokec5 from\cf4 \strokec4  tensorflow.keras.layers \cf5 \strokec5 import\cf4 \strokec4  \cf6 \strokec6 GRU\cf4 \strokec4 , Dense, Dropout\cb1 \
\cf5 \cb3 \strokec5 from\cf4 \strokec4  tensorflow.keras.callbacks \cf5 \strokec5 import\cf4 \strokec4  EarlyStopping\cb1 \
\cf5 \cb3 \strokec5 import\cf4 \strokec4  requests\cb1 \
\cf5 \cb3 \strokec5 import\cf4 \strokec4  time\cb1 \
\cf5 \cb3 \strokec5 from\cf4 \strokec4  datetime \cf5 \strokec5 import\cf4 \strokec4  datetime\cb1 \
\cf5 \cb3 \strokec5 import\cf4 \strokec4  yfinance \cf5 \strokec5 as\cf4 \strokec4  yf\cb1 \
\cf5 \cb3 \strokec5 import\cf4 \strokec4  warnings\cb1 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 warnings.filterwarnings(\cf7 \strokec7 'ignore'\cf4 \strokec4 )\cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf2 \cb3 \strokec2 # ============================================\cf4 \cb1 \strokec4 \
\cf2 \cb3 \strokec2 # \uc0\u55357 \u56593  PAGE CONFIGURATION\cf4 \cb1 \strokec4 \
\cf2 \cb3 \strokec2 # ============================================\cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 st.set_page_config(\cb1 \
\cb3     \cf8 \strokec8 page_title\cf4 \strokec9 =\cf7 \strokec7 "Crypto Predictor"\cf4 \strokec4 ,\cb1 \
\cb3     \cf8 \strokec8 page_icon\cf4 \strokec9 =\cf7 \strokec7 "\uc0\u55357 \u56622 "\cf4 \strokec4 ,\cb1 \
\cb3     \cf8 \strokec8 layout\cf4 \strokec9 =\cf7 \strokec7 "wide"\cf4 \cb1 \strokec4 \
\cb3 )\cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf2 \cb3 \strokec2 # ============================================\cf4 \cb1 \strokec4 \
\cf2 \cb3 \strokec2 # \uc0\u55357 \u56593  SECTION 1: YOUR MANUAL INPUTS\cf4 \cb1 \strokec4 \
\cf2 \cb3 \strokec2 # ============================================\cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf6 \cb3 \strokec6 BANKROLL\cf4 \strokec4  \strokec9 =\strokec4  \cf10 \strokec10 100.00\cf4 \cb1 \strokec4 \
\cf6 \cb3 \strokec6 MAX_RISK_PER_TRADE\cf4 \strokec4  \strokec9 =\strokec4  \cf10 \strokec10 0.02\cf4 \cb1 \strokec4 \
\cf6 \cb3 \strokec6 MIN_EDGE\cf4 \strokec4  \strokec9 =\strokec4  \cf10 \strokec10 0.05\cf4 \cb1 \strokec4 \
\cf6 \cb3 \strokec6 PREDICT_WINDOW\cf4 \strokec4  \strokec9 =\strokec4  \cf10 \strokec10 5\cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf2 \cb3 \strokec2 # Yahoo Finance symbols and CoinPaprika IDs\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf6 \cb3 \strokec6 COINS\cf4 \strokec4  \strokec9 =\strokec4  [\cb1 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3     \{\cf7 \strokec7 'yfinance'\cf4 \strokec4 : \cf7 \strokec7 'BTC-USD'\cf4 \strokec4 , \cf7 \strokec7 'paprika'\cf4 \strokec4 : \cf7 \strokec7 'btc-bitcoin'\cf4 \strokec4 \},\cb1 \
\cb3     \{\cf7 \strokec7 'yfinance'\cf4 \strokec4 : \cf7 \strokec7 'ETH-USD'\cf4 \strokec4 , \cf7 \strokec7 'paprika'\cf4 \strokec4 : \cf7 \strokec7 'eth-ethereum'\cf4 \strokec4 \},\cb1 \
\cb3     \{\cf7 \strokec7 'yfinance'\cf4 \strokec4 : \cf7 \strokec7 'SOL-USD'\cf4 \strokec4 , \cf7 \strokec7 'paprika'\cf4 \strokec4 : \cf7 \strokec7 'sol-solana'\cf4 \strokec4 \},\cb1 \
\cb3     \{\cf7 \strokec7 'yfinance'\cf4 \strokec4 : \cf7 \strokec7 'BNB-USD'\cf4 \strokec4 , \cf7 \strokec7 'paprika'\cf4 \strokec4 : \cf7 \strokec7 'bnb-binance-coin'\cf4 \strokec4 \},\cb1 \
\cb3     \{\cf7 \strokec7 'yfinance'\cf4 \strokec4 : \cf7 \strokec7 'XRP-USD'\cf4 \strokec4 , \cf7 \strokec7 'paprika'\cf4 \strokec4 : \cf7 \strokec7 'xrp-xrp'\cf4 \strokec4 \},\cb1 \
\cb3     \{\cf7 \strokec7 'yfinance'\cf4 \strokec4 : \cf7 \strokec7 'DOGE-USD'\cf4 \strokec4 , \cf7 \strokec7 'paprika'\cf4 \strokec4 : \cf7 \strokec7 'doge-dogecoin'\cf4 \strokec4 \},\cb1 \
\cb3 ]\cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf6 \cb3 \strokec6 USE_DUNE\cf4 \strokec4  \strokec9 =\strokec4  \cf11 \strokec11 False\cf4 \cb1 \strokec4 \
\cf6 \cb3 \strokec6 DUNE_API_KEY\cf4 \strokec4  \strokec9 =\strokec4  \cf7 \strokec7 ""\cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf2 \cb3 \strokec2 # ============================================\cf4 \cb1 \strokec4 \
\cf2 \cb3 \strokec2 # \uc0\u55357 \u56522  CACHED DATA FUNCTIONS\cf4 \cb1 \strokec4 \
\cf2 \cb3 \strokec2 # ============================================\cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf12 \cb3 \strokec12 @st.cache_data\cf4 \strokec4 (\cf8 \strokec8 ttl\cf4 \strokec9 =\cf10 \strokec10 60\cf4 \strokec4 )\cb1 \
\pard\pardeftab720\partightenfactor0
\cf13 \cb3 \strokec13 def\cf4 \strokec4  \cf12 \strokec12 fetch_yahoo_data\cf4 \strokec4 (\cf8 \strokec8 symbol\cf4 \strokec4 ):\cb1 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3     \cf7 \strokec7 """Fetch OHLCV data from Yahoo Finance (cached)"""\cf4 \cb1 \strokec4 \
\cb3     \cf5 \strokec5 try\cf4 \strokec4 :\cb1 \
\cb3         ticker \strokec9 =\strokec4  yf.Ticker(symbol)\cb1 \
\cb3         df \strokec9 =\strokec4  ticker.history(\cf8 \strokec8 period\cf4 \strokec9 =\cf7 \strokec7 '5d'\cf4 \strokec4 , \cf8 \strokec8 interval\cf4 \strokec9 =\cf7 \strokec7 '1m'\cf4 \strokec4 )\cb1 \
\cb3         \cb1 \
\cb3         \cf5 \strokec5 if\cf4 \strokec4  df.empty:\cb1 \
\cb3             \cf5 \strokec5 return\cf4 \strokec4  pd.DataFrame()\cb1 \
\cb3         \cb1 \
\cb3         df \strokec9 =\strokec4  df.reset_index()\cb1 \
\cb3         df \strokec9 =\strokec4  df.rename(\cf8 \strokec8 columns\cf4 \strokec9 =\strokec4 \{\cb1 \
\cb3             \cf7 \strokec7 'Datetime'\cf4 \strokec4 : \cf7 \strokec7 'time'\cf4 \strokec4 ,\cb1 \
\cb3             \cf7 \strokec7 'Open'\cf4 \strokec4 : \cf7 \strokec7 'open'\cf4 \strokec4 ,\cb1 \
\cb3             \cf7 \strokec7 'High'\cf4 \strokec4 : \cf7 \strokec7 'high'\cf4 \strokec4 ,\cb1 \
\cb3             \cf7 \strokec7 'Low'\cf4 \strokec4 : \cf7 \strokec7 'low'\cf4 \strokec4 ,\cb1 \
\cb3             \cf7 \strokec7 'Close'\cf4 \strokec4 : \cf7 \strokec7 'close'\cf4 \strokec4 ,\cb1 \
\cb3             \cf7 \strokec7 'Volume'\cf4 \strokec4 : \cf7 \strokec7 'volume'\cf4 \cb1 \strokec4 \
\cb3         \})\cb1 \
\cb3         \cb1 \
\cb3         \cf5 \strokec5 return\cf4 \strokec4  df\cb1 \
\cb3     \cf5 \strokec5 except\cf4 \strokec4  \cf14 \strokec14 Exception\cf4 \strokec4  \cf5 \strokec5 as\cf4 \strokec4  e:\cb1 \
\cb3         \cf5 \strokec5 return\cf4 \strokec4  pd.DataFrame()\cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf12 \cb3 \strokec12 @st.cache_data\cf4 \strokec4 (\cf8 \strokec8 ttl\cf4 \strokec9 =\cf10 \strokec10 300\cf4 \strokec4 )\cb1 \
\pard\pardeftab720\partightenfactor0
\cf13 \cb3 \strokec13 def\cf4 \strokec4  \cf12 \strokec12 fetch_coinpaprika_data\cf4 \strokec4 (\cf8 \strokec8 coin_id\cf4 \strokec4 ):\cb1 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3     \cf7 \strokec7 """Fetch data from CoinPaprika (cached for 5 minutes)"""\cf4 \cb1 \strokec4 \
\cb3     \cf5 \strokec5 try\cf4 \strokec4 :\cb1 \
\cb3         url \strokec9 =\strokec4  \cf13 \strokec13 f\cf7 \strokec7 "https://api.coinpaprika.com/v1/tickers/\cf13 \strokec13 \{\cf4 \strokec4 coin_id\cf13 \strokec13 \}\cf7 \strokec7 "\cf4 \cb1 \strokec4 \
\cb3         response \strokec9 =\strokec4  requests.get(url, \cf8 \strokec8 timeout\cf4 \strokec9 =\cf10 \strokec10 5\cf4 \strokec4 )\cb1 \
\cb3         data \strokec9 =\strokec4  response.json()\cb1 \
\cb3         \cb1 \
\cb3         quotes \strokec9 =\strokec4  data.get(\cf7 \strokec7 'quotes'\cf4 \strokec4 , \{\}).get(\cf7 \strokec7 'USD'\cf4 \strokec4 , \{\})\cb1 \
\cb3         \cb1 \
\cb3         \cf5 \strokec5 return\cf4 \strokec4  \{\cb1 \
\cb3             \cf7 \strokec7 'price'\cf4 \strokec4 : quotes.get(\cf7 \strokec7 'price'\cf4 \strokec4 , \cf10 \strokec10 0\cf4 \strokec4 ),\cb1 \
\cb3             \cf7 \strokec7 'volume_24h'\cf4 \strokec4 : quotes.get(\cf7 \strokec7 'volume_24h'\cf4 \strokec4 , \cf10 \strokec10 0\cf4 \strokec4 ),\cb1 \
\cb3             \cf7 \strokec7 'market_cap'\cf4 \strokec4 : quotes.get(\cf7 \strokec7 'market_cap'\cf4 \strokec4 , \cf10 \strokec10 0\cf4 \strokec4 ),\cb1 \
\cb3             \cf7 \strokec7 'percent_change_15m'\cf4 \strokec4 : quotes.get(\cf7 \strokec7 'percent_change_15m'\cf4 \strokec4 , \cf10 \strokec10 0\cf4 \strokec4 ),\cb1 \
\cb3             \cf7 \strokec7 'percent_change_1h'\cf4 \strokec4 : quotes.get(\cf7 \strokec7 'percent_change_1h'\cf4 \strokec4 , \cf10 \strokec10 0\cf4 \strokec4 ),\cb1 \
\cb3             \cf7 \strokec7 'percent_change_24h'\cf4 \strokec4 : quotes.get(\cf7 \strokec7 'percent_change_24h'\cf4 \strokec4 , \cf10 \strokec10 0\cf4 \strokec4 ),\cb1 \
\cb3             \cf7 \strokec7 'ath_price'\cf4 \strokec4 : quotes.get(\cf7 \strokec7 'ath_price'\cf4 \strokec4 , \cf10 \strokec10 0\cf4 \strokec4 ),\cb1 \
\cb3             \cf7 \strokec7 'total_supply'\cf4 \strokec4 : data.get(\cf7 \strokec7 'total_supply'\cf4 \strokec4 , \cf10 \strokec10 0\cf4 \strokec4 )\cb1 \
\cb3         \}\cb1 \
\cb3     \cf5 \strokec5 except\cf4 \strokec4 :\cb1 \
\cb3         \cf5 \strokec5 return\cf4 \strokec4  \{\cb1 \
\cb3             \cf7 \strokec7 'price'\cf4 \strokec4 : \cf10 \strokec10 0\cf4 \strokec4 , \cf7 \strokec7 'volume_24h'\cf4 \strokec4 : \cf10 \strokec10 0\cf4 \strokec4 , \cf7 \strokec7 'market_cap'\cf4 \strokec4 : \cf10 \strokec10 0\cf4 \strokec4 ,\cb1 \
\cb3             \cf7 \strokec7 'percent_change_15m'\cf4 \strokec4 : \cf10 \strokec10 0\cf4 \strokec4 , \cf7 \strokec7 'percent_change_1h'\cf4 \strokec4 : \cf10 \strokec10 0\cf4 \strokec4 ,\cb1 \
\cb3             \cf7 \strokec7 'percent_change_24h'\cf4 \strokec4 : \cf10 \strokec10 0\cf4 \strokec4 , \cf7 \strokec7 'ath_price'\cf4 \strokec4 : \cf10 \strokec10 0\cf4 \strokec4 , \cf7 \strokec7 'total_supply'\cf4 \strokec4 : \cf10 \strokec10 0\cf4 \cb1 \strokec4 \
\cb3         \}\cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf12 \cb3 \strokec12 @st.cache_data\cf4 \strokec4 (\cf8 \strokec8 ttl\cf4 \strokec9 =\cf10 \strokec10 300\cf4 \strokec4 )\cb1 \
\pard\pardeftab720\partightenfactor0
\cf13 \cb3 \strokec13 def\cf4 \strokec4  \cf12 \strokec12 fetch_coinpaprika_twitter\cf4 \strokec4 (\cf8 \strokec8 coin_id\cf4 \strokec4 ):\cb1 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3     \cf7 \strokec7 """Fetch Twitter stats from CoinPaprika"""\cf4 \cb1 \strokec4 \
\cb3     \cf5 \strokec5 try\cf4 \strokec4 :\cb1 \
\cb3         url \strokec9 =\strokec4  \cf13 \strokec13 f\cf7 \strokec7 "https://api.coinpaprika.com/v1/coins/\cf13 \strokec13 \{\cf4 \strokec4 coin_id\cf13 \strokec13 \}\cf7 \strokec7 /twitter"\cf4 \cb1 \strokec4 \
\cb3         response \strokec9 =\strokec4  requests.get(url, \cf8 \strokec8 timeout\cf4 \strokec9 =\cf10 \strokec10 5\cf4 \strokec4 )\cb1 \
\cb3         data \strokec9 =\strokec4  response.json()\cb1 \
\cb3         \cf5 \strokec5 return\cf4 \strokec4  \{\cb1 \
\cb3             \cf7 \strokec7 'twitter_followers'\cf4 \strokec4 : data.get(\cf7 \strokec7 'followers'\cf4 \strokec4 , \cf10 \strokec10 0\cf4 \strokec4 ),\cb1 \
\cb3             \cf7 \strokec7 'twitter_statuses'\cf4 \strokec4 : data.get(\cf7 \strokec7 'statuses'\cf4 \strokec4 , \cf10 \strokec10 0\cf4 \strokec4 )\cb1 \
\cb3         \}\cb1 \
\cb3     \cf5 \strokec5 except\cf4 \strokec4 :\cb1 \
\cb3         \cf5 \strokec5 return\cf4 \strokec4  \{\cf7 \strokec7 'twitter_followers'\cf4 \strokec4 : \cf10 \strokec10 0\cf4 \strokec4 , \cf7 \strokec7 'twitter_statuses'\cf4 \strokec4 : \cf10 \strokec10 0\cf4 \strokec4 \}\cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf12 \cb3 \strokec12 @st.cache_data\cf4 \strokec4 (\cf8 \strokec8 ttl\cf4 \strokec9 =\cf10 \strokec10 10\cf4 \strokec4 )\cb1 \
\pard\pardeftab720\partightenfactor0
\cf13 \cb3 \strokec13 def\cf4 \strokec4  \cf12 \strokec12 fetch_kalshi_orderbook\cf4 \strokec4 (\cf8 \strokec8 ticker\cf4 \strokec4 ):\cb1 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3     \cf7 \strokec7 """Fetch live order book from Kalshi (cached 10 seconds)"""\cf4 \cb1 \strokec4 \
\cb3     \cf5 \strokec5 try\cf4 \strokec4 :\cb1 \
\cb3         url \strokec9 =\strokec4  \cf13 \strokec13 f\cf7 \strokec7 "https://external-api.kalshi.com/trade-api/v2/markets/\cf13 \strokec13 \{\cf4 \strokec4 ticker\cf13 \strokec13 \}\cf7 \strokec7 /orderbook"\cf4 \cb1 \strokec4 \
\cb3         response \strokec9 =\strokec4  requests.get(url, \cf8 \strokec8 timeout\cf4 \strokec9 =\cf10 \strokec10 5\cf4 \strokec4 )\cb1 \
\cb3         data \strokec9 =\strokec4  response.json()\cb1 \
\cb3         book \strokec9 =\strokec4  data.get(\cf7 \strokec7 'orderbook_fp'\cf4 \strokec4 , \{\})\cb1 \
\cb3         \cb1 \
\cb3         yes_bids \strokec9 =\strokec4  book.get(\cf7 \strokec7 'yes_dollars'\cf4 \strokec4 , [])\cb1 \
\cb3         no_bids \strokec9 =\strokec4  book.get(\cf7 \strokec7 'no_dollars'\cf4 \strokec4 , [])\cb1 \
\cb3         \cb1 \
\cb3         metrics \strokec9 =\strokec4  \{\cb1 \
\cb3             \cf7 \strokec7 'best_yes_bid'\cf4 \strokec4 : \cf14 \strokec14 float\cf4 \strokec4 (yes_bids[\cf10 \strokec10 0\cf4 \strokec4 ][\cf10 \strokec10 0\cf4 \strokec4 ]) \cf5 \strokec5 if\cf4 \strokec4  yes_bids \cf5 \strokec5 else\cf4 \strokec4  \cf10 \strokec10 0\cf4 \strokec4 ,\cb1 \
\cb3             \cf7 \strokec7 'best_no_bid'\cf4 \strokec4 : \cf14 \strokec14 float\cf4 \strokec4 (no_bids[\cf10 \strokec10 0\cf4 \strokec4 ][\cf10 \strokec10 0\cf4 \strokec4 ]) \cf5 \strokec5 if\cf4 \strokec4  no_bids \cf5 \strokec5 else\cf4 \strokec4  \cf10 \strokec10 0\cf4 \strokec4 ,\cb1 \
\cb3             \cf7 \strokec7 'yes_volume'\cf4 \strokec4 : \cf15 \strokec15 sum\cf4 \strokec4 ([\cf14 \strokec14 float\cf4 \strokec4 (p[\cf10 \strokec10 1\cf4 \strokec4 ]) \cf5 \strokec5 for\cf4 \strokec4  p \cf5 \strokec5 in\cf4 \strokec4  yes_bids[:\cf10 \strokec10 10\cf4 \strokec4 ]]) \cf5 \strokec5 if\cf4 \strokec4  yes_bids \cf5 \strokec5 else\cf4 \strokec4  \cf10 \strokec10 0\cf4 \strokec4 ,\cb1 \
\cb3             \cf7 \strokec7 'no_volume'\cf4 \strokec4 : \cf15 \strokec15 sum\cf4 \strokec4 ([\cf14 \strokec14 float\cf4 \strokec4 (p[\cf10 \strokec10 1\cf4 \strokec4 ]) \cf5 \strokec5 for\cf4 \strokec4  p \cf5 \strokec5 in\cf4 \strokec4  no_bids[:\cf10 \strokec10 10\cf4 \strokec4 ]]) \cf5 \strokec5 if\cf4 \strokec4  no_bids \cf5 \strokec5 else\cf4 \strokec4  \cf10 \strokec10 0\cf4 \strokec4 ,\cb1 \
\cb3         \}\cb1 \
\cb3         \cb1 \
\cb3         metrics[\cf7 \strokec7 'spread'\cf4 \strokec4 ] \strokec9 =\strokec4  metrics[\cf7 \strokec7 'best_yes_bid'\cf4 \strokec4 ] \strokec9 -\strokec4  metrics[\cf7 \strokec7 'best_no_bid'\cf4 \strokec4 ]\cb1 \
\cb3         total_volume \strokec9 =\strokec4  metrics[\cf7 \strokec7 'yes_volume'\cf4 \strokec4 ] \strokec9 +\strokec4  metrics[\cf7 \strokec7 'no_volume'\cf4 \strokec4 ]\cb1 \
\cb3         metrics[\cf7 \strokec7 'imbalance'\cf4 \strokec4 ] \strokec9 =\strokec4  (metrics[\cf7 \strokec7 'yes_volume'\cf4 \strokec4 ] \strokec9 -\strokec4  metrics[\cf7 \strokec7 'no_volume'\cf4 \strokec4 ]) \strokec9 /\strokec4  (total_volume \strokec9 +\strokec4  \cf10 \strokec10 1\cf4 \strokec4 )\cb1 \
\cb3         \cb1 \
\cb3         \cf5 \strokec5 return\cf4 \strokec4  metrics\cb1 \
\cb3     \cf5 \strokec5 except\cf4 \strokec4 :\cb1 \
\cb3         \cf5 \strokec5 return\cf4 \strokec4  \{\cf7 \strokec7 'best_yes_bid'\cf4 \strokec4 : \cf10 \strokec10 0\cf4 \strokec4 , \cf7 \strokec7 'best_no_bid'\cf4 \strokec4 : \cf10 \strokec10 0\cf4 \strokec4 , \cf7 \strokec7 'spread'\cf4 \strokec4 : \cf10 \strokec10 0\cf4 \strokec4 , \cf7 \strokec7 'imbalance'\cf4 \strokec4 : \cf10 \strokec10 0\cf4 \strokec4 \}\cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf2 \cb3 \strokec2 # ============================================\cf4 \cb1 \strokec4 \
\cf2 \cb3 \strokec2 # \uc0\u55357 \u56522  TECHNICAL INDICATORS & MODEL\cf4 \cb1 \strokec4 \
\cf2 \cb3 \strokec2 # ============================================\cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf13 \cb3 \strokec13 def\cf4 \strokec4  \cf12 \strokec12 add_technical_indicators\cf4 \strokec4 (\cf8 \strokec8 df\cf4 \strokec4 ):\cb1 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3     \cf7 \strokec7 """Add technical indicators"""\cf4 \cb1 \strokec4 \
\cb3     df[\cf7 \strokec7 'sma_5'\cf4 \strokec4 ] \strokec9 =\strokec4  df[\cf7 \strokec7 'close'\cf4 \strokec4 ].rolling(\cf10 \strokec10 5\cf4 \strokec4 ).mean()\cb1 \
\cb3     df[\cf7 \strokec7 'sma_10'\cf4 \strokec4 ] \strokec9 =\strokec4  df[\cf7 \strokec7 'close'\cf4 \strokec4 ].rolling(\cf10 \strokec10 10\cf4 \strokec4 ).mean()\cb1 \
\cb3     \cb1 \
\cb3     delta \strokec9 =\strokec4  df[\cf7 \strokec7 'close'\cf4 \strokec4 ].diff()\cb1 \
\cb3     gain \strokec9 =\strokec4  (delta.where(delta \strokec9 >\strokec4  \cf10 \strokec10 0\cf4 \strokec4 , \cf10 \strokec10 0\cf4 \strokec4 )).rolling(\cf10 \strokec10 14\cf4 \strokec4 ).mean()\cb1 \
\cb3     loss \strokec9 =\strokec4  (\strokec9 -\strokec4 delta.where(delta \strokec9 <\strokec4  \cf10 \strokec10 0\cf4 \strokec4 , \cf10 \strokec10 0\cf4 \strokec4 )).rolling(\cf10 \strokec10 14\cf4 \strokec4 ).mean()\cb1 \
\cb3     rs \strokec9 =\strokec4  gain \strokec9 /\strokec4  loss\cb1 \
\cb3     df[\cf7 \strokec7 'rsi'\cf4 \strokec4 ] \strokec9 =\strokec4  \cf10 \strokec10 100\cf4 \strokec4  \strokec9 -\strokec4  (\cf10 \strokec10 100\cf4 \strokec4  \strokec9 /\strokec4  (\cf10 \strokec10 1\cf4 \strokec4  \strokec9 +\strokec4  rs))\cb1 \
\cb3     \cb1 \
\cb3     df[\cf7 \strokec7 'return_1'\cf4 \strokec4 ] \strokec9 =\strokec4  df[\cf7 \strokec7 'close'\cf4 \strokec4 ].pct_change()\cb1 \
\cb3     df[\cf7 \strokec7 'return_5'\cf4 \strokec4 ] \strokec9 =\strokec4  df[\cf7 \strokec7 'close'\cf4 \strokec4 ].pct_change(\cf10 \strokec10 5\cf4 \strokec4 )\cb1 \
\cb3     df[\cf7 \strokec7 'price_range'\cf4 \strokec4 ] \strokec9 =\strokec4  (df[\cf7 \strokec7 'high'\cf4 \strokec4 ] \strokec9 -\strokec4  df[\cf7 \strokec7 'low'\cf4 \strokec4 ]) \strokec9 /\strokec4  df[\cf7 \strokec7 'close'\cf4 \strokec4 ]\cb1 \
\cb3     df[\cf7 \strokec7 'volume_ratio'\cf4 \strokec4 ] \strokec9 =\strokec4  df[\cf7 \strokec7 'volume'\cf4 \strokec4 ] \strokec9 /\strokec4  df[\cf7 \strokec7 'volume'\cf4 \strokec4 ].rolling(\cf10 \strokec10 10\cf4 \strokec4 ).mean()\cb1 \
\cb3     \cb1 \
\cb3     \cf5 \strokec5 return\cf4 \strokec4  df\cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf13 \cb3 \strokec13 def\cf4 \strokec4  \cf12 \strokec12 calculate_kelly_bet\cf4 \strokec4 (\cf8 \strokec8 win_prob\cf4 \strokec4 , \cf8 \strokec8 bankroll\cf4 \strokec4 , \cf8 \strokec8 max_risk_pct\cf4 \strokec9 =\cf10 \strokec10 0.02\cf4 \strokec4 ):\cb1 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3     \cf7 \strokec7 """Calculate optimal bet size using Kelly Criterion"""\cf4 \cb1 \strokec4 \
\cb3     odds \strokec9 =\strokec4  \cf10 \strokec10 1.0\cf4 \cb1 \strokec4 \
\cb3     p \strokec9 =\strokec4  win_prob\cb1 \
\cb3     q \strokec9 =\strokec4  \cf10 \strokec10 1\cf4 \strokec4  \strokec9 -\strokec4  p\cb1 \
\cb3     \cb1 \
\cb3     \cf5 \strokec5 if\cf4 \strokec4  p \strokec9 <=\strokec4  \cf10 \strokec10 0.5\cf4 \strokec4 :\cb1 \
\cb3         \cf5 \strokec5 return\cf4 \strokec4  \cf10 \strokec10 0\cf4 \cb1 \strokec4 \
\cb3     \cb1 \
\cb3     kelly_fraction \strokec9 =\strokec4  (p \strokec9 *\strokec4  odds \strokec9 -\strokec4  q) \strokec9 /\strokec4  odds\cb1 \
\cb3     kelly_fraction \strokec9 =\strokec4  \cf15 \strokec15 min\cf4 \strokec4 (kelly_fraction, max_risk_pct)\cb1 \
\cb3     \cb1 \
\cb3     \cf5 \strokec5 if\cf4 \strokec4  kelly_fraction \strokec9 <\strokec4  \cf10 \strokec10 0.01\cf4 \strokec4 :\cb1 \
\cb3         \cf5 \strokec5 return\cf4 \strokec4  \cf10 \strokec10 0\cf4 \cb1 \strokec4 \
\cb3     \cb1 \
\cb3     \cf5 \strokec5 return\cf4 \strokec4  kelly_fraction \strokec9 *\strokec4  bankroll\cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf13 \cb3 \strokec13 def\cf4 \strokec4  \cf12 \strokec12 build_gru_model\cf4 \strokec4 (\cf8 \strokec8 input_shape\cf4 \strokec4 ):\cb1 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3     \cf7 \strokec7 """Build the GRU model"""\cf4 \cb1 \strokec4 \
\cb3     model \strokec9 =\strokec4  Sequential([\cb1 \
\cb3         GRU(\cf10 \strokec10 64\cf4 \strokec4 , \cf8 \strokec8 input_shape\cf4 \strokec9 =\strokec4 input_shape, \cf8 \strokec8 return_sequences\cf4 \strokec9 =\cf11 \strokec11 True\cf4 \strokec4 ),\cb1 \
\cb3         Dropout(\cf10 \strokec10 0.2\cf4 \strokec4 ),\cb1 \
\cb3         GRU(\cf10 \strokec10 32\cf4 \strokec4 ),\cb1 \
\cb3         Dropout(\cf10 \strokec10 0.2\cf4 \strokec4 ),\cb1 \
\cb3         Dense(\cf10 \strokec10 16\cf4 \strokec4 , \cf8 \strokec8 activation\cf4 \strokec9 =\cf7 \strokec7 'relu'\cf4 \strokec4 ),\cb1 \
\cb3         Dense(\cf10 \strokec10 1\cf4 \strokec4 , \cf8 \strokec8 activation\cf4 \strokec9 =\cf7 \strokec7 'sigmoid'\cf4 \strokec4 )\cb1 \
\cb3     ])\cb1 \
\cb3     model.compile(\cf8 \strokec8 optimizer\cf4 \strokec9 =\cf7 \strokec7 'adam'\cf4 \strokec4 , \cf8 \strokec8 loss\cf4 \strokec9 =\cf7 \strokec7 'binary_crossentropy'\cf4 \strokec4 , \cf8 \strokec8 metrics\cf4 \strokec9 =\strokec4 [\cf7 \strokec7 'accuracy'\cf4 \strokec4 ])\cb1 \
\cb3     \cf5 \strokec5 return\cf4 \strokec4  model\cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf2 \cb3 \strokec2 # ============================================\cf4 \cb1 \strokec4 \
\cf2 \cb3 \strokec2 # \uc0\u55357 \u56960  MAIN DASHBOARD EXECUTION\cf4 \cb1 \strokec4 \
\cf2 \cb3 \strokec2 # ============================================\cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf6 \cb3 \strokec6 LOOKBACK_WINDOW\cf4 \strokec4  \strokec9 =\strokec4  \cf10 \strokec10 30\cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf2 \cb3 \strokec2 # --- Dashboard Header ---\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 st.title(\cf7 \strokec7 "\uc0\u55357 \u56622  Optimized Crypto Predictor"\cf4 \strokec4 )\cb1 \
\cb3 st.caption(\cf13 \strokec13 f\cf7 \strokec7 "Last updated: \cf13 \strokec13 \{\cf4 \strokec4 datetime.now().strftime(\cf7 \strokec7 '%Y-%m-\cf13 \strokec13 %d\cf7 \strokec7  %H:%M:%S'\cf4 \strokec4 )\cf13 \strokec13 \}\cf7 \strokec7 "\cf4 \strokec4 )\cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf2 \cb3 \strokec2 # --- Metrics Row ---\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 col1, col2, col3, col4, col5 \strokec9 =\strokec4  st.columns(\cf10 \strokec10 5\cf4 \strokec4 )\cb1 \
\cb3 col1.metric(\cf7 \strokec7 "Bankroll"\cf4 \strokec4 , \cf13 \strokec13 f\cf7 \strokec7 "$\cf13 \strokec13 \{\cf6 \strokec6 BANKROLL\cf13 \strokec13 :.2f\}\cf7 \strokec7 "\cf4 \strokec4 )\cb1 \
\cb3 col2.metric(\cf7 \strokec7 "Min Edge"\cf4 \strokec4 , \cf13 \strokec13 f\cf7 \strokec7 "\cf13 \strokec13 \{\cf6 \strokec6 MIN_EDGE\cf4 \strokec9 *\cf10 \strokec10 100\cf13 \strokec13 :.0f\}\cf7 \strokec7 %"\cf4 \strokec4 )\cb1 \
\cb3 col3.metric(\cf7 \strokec7 "Max Risk/Trade"\cf4 \strokec4 , \cf13 \strokec13 f\cf7 \strokec7 "\cf13 \strokec13 \{\cf6 \strokec6 MAX_RISK_PER_TRADE\cf4 \strokec9 *\cf10 \strokec10 100\cf13 \strokec13 :.0f\}\cf7 \strokec7 %"\cf4 \strokec4 )\cb1 \
\cb3 col4.metric(\cf7 \strokec7 "Predict Window"\cf4 \strokec4 , \cf13 \strokec13 f\cf7 \strokec7 "\cf13 \strokec13 \{\cf6 \strokec6 PREDICT_WINDOW\cf13 \strokec13 \}\cf7 \strokec7  min"\cf4 \strokec4 )\cb1 \
\cb3 col5.metric(\cf7 \strokec7 "Data Sources"\cf4 \strokec4 , \cf7 \strokec7 "Yahoo + Paprika"\cf4 \strokec4 )\cb1 \
\
\cb3 st.divider()\cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf2 \cb3 \strokec2 # --- Run Predictions ---\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 all_results \strokec9 =\strokec4  []\cb1 \
\cb3 best_bets \strokec9 =\strokec4  []\cb1 \
\
\cb3 progress_bar \strokec9 =\strokec4  st.progress(\cf10 \strokec10 0\cf4 \strokec4 )\cb1 \
\cb3 status_text \strokec9 =\strokec4  st.empty()\cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf5 \cb3 \strokec5 for\cf4 \strokec4  idx, coin \cf5 \strokec5 in\cf4 \strokec4  \cf15 \strokec15 enumerate\cf4 \strokec4 (\cf6 \strokec6 COINS\cf4 \strokec4 ):\cb1 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3     status_text.text(\cf13 \strokec13 f\cf7 \strokec7 "Processing \cf13 \strokec13 \{\cf4 \strokec4 coin[\cf7 \strokec7 'yfinance'\cf4 \strokec4 ]\cf13 \strokec13 \}\cf7 \strokec7 ..."\cf4 \strokec4 )\cb1 \
\cb3     \cb1 \
\cb3     \cf5 \strokec5 try\cf4 \strokec4 :\cb1 \
\cb3         \cf2 \strokec2 # Fetch data\cf4 \cb1 \strokec4 \
\cb3         df \strokec9 =\strokec4  fetch_yahoo_data(coin[\cf7 \strokec7 'yfinance'\cf4 \strokec4 ])\cb1 \
\cb3         \cb1 \
\cb3         \cf5 \strokec5 if\cf4 \strokec4  df.empty:\cb1 \
\cb3             \cf5 \strokec5 continue\cf4 \cb1 \strokec4 \
\cb3         \cb1 \
\cb3         df \strokec9 =\strokec4  add_technical_indicators(df)\cb1 \
\cb3         \cb1 \
\cb3         \cf2 \strokec2 # Fetch CoinPaprika data\cf4 \cb1 \strokec4 \
\cb3         paprika_data \strokec9 =\strokec4  fetch_coinpaprika_data(coin[\cf7 \strokec7 'paprika'\cf4 \strokec4 ])\cb1 \
\cb3         paprika_twitter \strokec9 =\strokec4  fetch_coinpaprika_twitter(coin[\cf7 \strokec7 'paprika'\cf4 \strokec4 ])\cb1 \
\cb3         \cb1 \
\cb3         df[\cf7 \strokec7 'paprika_price'\cf4 \strokec4 ] \strokec9 =\strokec4  paprika_data.get(\cf7 \strokec7 'price'\cf4 \strokec4 , \cf10 \strokec10 0\cf4 \strokec4 )\cb1 \
\cb3         df[\cf7 \strokec7 'paprika_volume'\cf4 \strokec4 ] \strokec9 =\strokec4  paprika_data.get(\cf7 \strokec7 'volume_24h'\cf4 \strokec4 , \cf10 \strokec10 0\cf4 \strokec4 )\cb1 \
\cb3         df[\cf7 \strokec7 'paprika_mcap'\cf4 \strokec4 ] \strokec9 =\strokec4  paprika_data.get(\cf7 \strokec7 'market_cap'\cf4 \strokec4 , \cf10 \strokec10 0\cf4 \strokec4 )\cb1 \
\cb3         df[\cf7 \strokec7 'change_15m'\cf4 \strokec4 ] \strokec9 =\strokec4  paprika_data.get(\cf7 \strokec7 'percent_change_15m'\cf4 \strokec4 , \cf10 \strokec10 0\cf4 \strokec4 )\cb1 \
\cb3         df[\cf7 \strokec7 'change_1h'\cf4 \strokec4 ] \strokec9 =\strokec4  paprika_data.get(\cf7 \strokec7 'percent_change_1h'\cf4 \strokec4 , \cf10 \strokec10 0\cf4 \strokec4 )\cb1 \
\cb3         df[\cf7 \strokec7 'change_24h'\cf4 \strokec4 ] \strokec9 =\strokec4  paprika_data.get(\cf7 \strokec7 'percent_change_24h'\cf4 \strokec4 , \cf10 \strokec10 0\cf4 \strokec4 )\cb1 \
\cb3         df[\cf7 \strokec7 'twitter_followers'\cf4 \strokec4 ] \strokec9 =\strokec4  paprika_twitter.get(\cf7 \strokec7 'twitter_followers'\cf4 \strokec4 , \cf10 \strokec10 0\cf4 \strokec4 )\cb1 \
\cb3         \cb1 \
\cb3         \cf2 \strokec2 # Fetch Kalshi order book\cf4 \cb1 \strokec4 \
\cb3         coin_name \strokec9 =\strokec4  coin[\cf7 \strokec7 'yfinance'\cf4 \strokec4 ].replace(\cf7 \strokec7 '-USD'\cf4 \strokec4 , \cf7 \strokec7 ''\cf4 \strokec4 )\cb1 \
\cb3         orderbook \strokec9 =\strokec4  fetch_kalshi_orderbook(\cf13 \strokec13 f\cf7 \strokec7 "KX\cf13 \strokec13 \{\cf4 \strokec4 coin_name\cf13 \strokec13 \}\cf7 \strokec7 "\cf4 \strokec4 )\cb1 \
\cb3         df[\cf7 \strokec7 'spread'\cf4 \strokec4 ] \strokec9 =\strokec4  orderbook.get(\cf7 \strokec7 'spread'\cf4 \strokec4 , \cf10 \strokec10 0\cf4 \strokec4 )\cb1 \
\cb3         df[\cf7 \strokec7 'imbalance'\cf4 \strokec4 ] \strokec9 =\strokec4  orderbook.get(\cf7 \strokec7 'imbalance'\cf4 \strokec4 , \cf10 \strokec10 0\cf4 \strokec4 )\cb1 \
\cb3         df[\cf7 \strokec7 'best_yes_bid'\cf4 \strokec4 ] \strokec9 =\strokec4  orderbook.get(\cf7 \strokec7 'best_yes_bid'\cf4 \strokec4 , \cf10 \strokec10 0\cf4 \strokec4 )\cb1 \
\cb3         \cb1 \
\cb3         \cf2 \strokec2 # Prepare data\cf4 \cb1 \strokec4 \
\cb3         df_clean \strokec9 =\strokec4  df.dropna()\cb1 \
\cb3         \cb1 \
\cb3         \cf5 \strokec5 if\cf4 \strokec4  \cf15 \strokec15 len\cf4 \strokec4 (df_clean) \strokec9 <\strokec4  \cf10 \strokec10 50\cf4 \strokec4 :\cb1 \
\cb3             \cf5 \strokec5 continue\cf4 \cb1 \strokec4 \
\cb3         \cb1 \
\cb3         \cf2 \strokec2 # Create target\cf4 \cb1 \strokec4 \
\cb3         df_clean[\cf7 \strokec7 'future_price'\cf4 \strokec4 ] \strokec9 =\strokec4  df_clean[\cf7 \strokec7 'close'\cf4 \strokec4 ].shift(\strokec9 -\cf6 \strokec6 PREDICT_WINDOW\cf4 \strokec4 )\cb1 \
\cb3         df_clean[\cf7 \strokec7 'target'\cf4 \strokec4 ] \strokec9 =\strokec4  (df_clean[\cf7 \strokec7 'future_price'\cf4 \strokec4 ] \strokec9 >\strokec4  df_clean[\cf7 \strokec7 'close'\cf4 \strokec4 ]).astype(\cf14 \strokec14 int\cf4 \strokec4 )\cb1 \
\cb3         df_clean \strokec9 =\strokec4  df_clean.dropna()\cb1 \
\cb3         \cb1 \
\cb3         \cf5 \strokec5 if\cf4 \strokec4  \cf15 \strokec15 len\cf4 \strokec4 (df_clean) \strokec9 <\strokec4  \cf10 \strokec10 50\cf4 \strokec4 :\cb1 \
\cb3             \cf5 \strokec5 continue\cf4 \cb1 \strokec4 \
\cb3         \cb1 \
\cb3         \cf2 \strokec2 # Feature engineering\cf4 \cb1 \strokec4 \
\cb3         feature_cols \strokec9 =\strokec4  [\cb1 \
\cb3             \cf7 \strokec7 'close'\cf4 \strokec4 , \cf7 \strokec7 'volume'\cf4 \strokec4 , \cf7 \strokec7 'return_1'\cf4 \strokec4 , \cf7 \strokec7 'return_5'\cf4 \strokec4 , \cf7 \strokec7 'price_range'\cf4 \strokec4 ,\cb1 \
\cb3             \cf7 \strokec7 'volume_ratio'\cf4 \strokec4 , \cf7 \strokec7 'rsi'\cf4 \strokec4 , \cf7 \strokec7 'sma_5'\cf4 \strokec4 , \cf7 \strokec7 'sma_10'\cf4 \strokec4 ,\cb1 \
\cb3             \cf7 \strokec7 'spread'\cf4 \strokec4 , \cf7 \strokec7 'imbalance'\cf4 \strokec4 , \cf7 \strokec7 'best_yes_bid'\cf4 \strokec4 ,\cb1 \
\cb3             \cf7 \strokec7 'paprika_price'\cf4 \strokec4 , \cf7 \strokec7 'paprika_volume'\cf4 \strokec4 , \cf7 \strokec7 'paprika_mcap'\cf4 \strokec4 ,\cb1 \
\cb3             \cf7 \strokec7 'change_15m'\cf4 \strokec4 , \cf7 \strokec7 'change_1h'\cf4 \strokec4 , \cf7 \strokec7 'change_24h'\cf4 \strokec4 ,\cb1 \
\cb3             \cf7 \strokec7 'twitter_followers'\cf4 \cb1 \strokec4 \
\cb3         ]\cb1 \
\cb3         \cb1 \
\cb3         available_cols \strokec9 =\strokec4  [col \cf5 \strokec5 for\cf4 \strokec4  col \cf5 \strokec5 in\cf4 \strokec4  feature_cols \cf5 \strokec5 if\cf4 \strokec4  col \cf11 \strokec11 in\cf4 \strokec4  df_clean.columns]\cb1 \
\cb3         X \strokec9 =\strokec4  df_clean[available_cols].values\cb1 \
\cb3         y \strokec9 =\strokec4  df_clean[\cf7 \strokec7 'target'\cf4 \strokec4 ].values\cb1 \
\cb3         \cb1 \
\cb3         \cf2 \strokec2 # Scale\cf4 \cb1 \strokec4 \
\cb3         scaler \strokec9 =\strokec4  MinMaxScaler()\cb1 \
\cb3         X_scaled \strokec9 =\strokec4  scaler.fit_transform(X)\cb1 \
\cb3         \cb1 \
\cb3         \cf2 \strokec2 # Create sequences\cf4 \cb1 \strokec4 \
\cb3         X_seq, y_seq \strokec9 =\strokec4  [], []\cb1 \
\cb3         \cf5 \strokec5 for\cf4 \strokec4  i \cf5 \strokec5 in\cf4 \strokec4  \cf15 \strokec15 range\cf4 \strokec4 (\cf6 \strokec6 LOOKBACK_WINDOW\cf4 \strokec4 , \cf15 \strokec15 len\cf4 \strokec4 (X_scaled)):\cb1 \
\cb3             X_seq.append(X_scaled[i\strokec9 -\cf6 \strokec6 LOOKBACK_WINDOW\cf4 \strokec4 :i])\cb1 \
\cb3             y_seq.append(y[i])\cb1 \
\cb3         \cb1 \
\cb3         X_seq \strokec9 =\strokec4  np.array(X_seq)\cb1 \
\cb3         y_seq \strokec9 =\strokec4  np.array(y_seq)\cb1 \
\cb3         \cb1 \
\cb3         \cf5 \strokec5 if\cf4 \strokec4  \cf15 \strokec15 len\cf4 \strokec4 (X_seq) \strokec9 <\strokec4  \cf10 \strokec10 30\cf4 \strokec4 :\cb1 \
\cb3             \cf5 \strokec5 continue\cf4 \cb1 \strokec4 \
\cb3         \cb1 \
\cb3         \cf2 \strokec2 # Train model\cf4 \cb1 \strokec4 \
\cb3         split_idx \strokec9 =\strokec4  \cf14 \strokec14 int\cf4 \strokec4 (\cf10 \strokec10 0.8\cf4 \strokec4  \strokec9 *\strokec4  \cf15 \strokec15 len\cf4 \strokec4 (X_seq))\cb1 \
\cb3         X_train, X_test \strokec9 =\strokec4  X_seq[:split_idx], X_seq[split_idx:]\cb1 \
\cb3         y_train, y_test \strokec9 =\strokec4  y_seq[:split_idx], y_seq[split_idx:]\cb1 \
\cb3         \cb1 \
\cb3         model \strokec9 =\strokec4  build_gru_model((\cf6 \strokec6 LOOKBACK_WINDOW\cf4 \strokec4 , \cf15 \strokec15 len\cf4 \strokec4 (available_cols)))\cb1 \
\cb3         early_stop \strokec9 =\strokec4  EarlyStopping(\cf8 \strokec8 monitor\cf4 \strokec9 =\cf7 \strokec7 'val_loss'\cf4 \strokec4 , \cf8 \strokec8 patience\cf4 \strokec9 =\cf10 \strokec10 5\cf4 \strokec4 , \cf8 \strokec8 restore_best_weights\cf4 \strokec9 =\cf11 \strokec11 True\cf4 \strokec4 )\cb1 \
\cb3         \cb1 \
\cb3         model.fit(X_train, y_train, \cf8 \strokec8 epochs\cf4 \strokec9 =\cf10 \strokec10 30\cf4 \strokec4 , \cf8 \strokec8 batch_size\cf4 \strokec9 =\cf10 \strokec10 16\cf4 \strokec4 , \cb1 \
\cb3                  \cf8 \strokec8 validation_split\cf4 \strokec9 =\cf10 \strokec10 0.2\cf4 \strokec4 , \cf8 \strokec8 callbacks\cf4 \strokec9 =\strokec4 [early_stop], \cf8 \strokec8 verbose\cf4 \strokec9 =\cf10 \strokec10 0\cf4 \strokec4 )\cb1 \
\cb3         \cb1 \
\cb3         \cf2 \strokec2 # Evaluate\cf4 \cb1 \strokec4 \
\cb3         test_loss, test_accuracy \strokec9 =\strokec4  model.evaluate(X_test, y_test, \cf8 \strokec8 verbose\cf4 \strokec9 =\cf10 \strokec10 0\cf4 \strokec4 )\cb1 \
\cb3         \cb1 \
\cb3         \cf2 \strokec2 # Live prediction\cf4 \cb1 \strokec4 \
\cb3         last_sequence \strokec9 =\strokec4  X_scaled[\strokec9 -\cf6 \strokec6 LOOKBACK_WINDOW\cf4 \strokec4 :].reshape(\cf10 \strokec10 1\cf4 \strokec4 , \cf6 \strokec6 LOOKBACK_WINDOW\cf4 \strokec4 , \cf15 \strokec15 len\cf4 \strokec4 (available_cols))\cb1 \
\cb3         win_prob \strokec9 =\strokec4  model.predict(last_sequence, \cf8 \strokec8 verbose\cf4 \strokec9 =\cf10 \strokec10 0\cf4 \strokec4 )[\cf10 \strokec10 0\cf4 \strokec4 ][\cf10 \strokec10 0\cf4 \strokec4 ]\cb1 \
\cb3         \cb1 \
\cb3         \cf2 \strokec2 # Calculate bet\cf4 \cb1 \strokec4 \
\cb3         bet_amount \strokec9 =\strokec4  calculate_kelly_bet(win_prob, \cf6 \strokec6 BANKROLL\cf4 \strokec4 , \cf6 \strokec6 MAX_RISK_PER_TRADE\cf4 \strokec4 )\cb1 \
\cb3         current_price \strokec9 =\strokec4  df_clean[\cf7 \strokec7 'close'\cf4 \strokec4 ].iloc[\strokec9 -\cf10 \strokec10 1\cf4 \strokec4 ]\cb1 \
\cb3         \cb1 \
\cb3         \cf2 \strokec2 # Determine action\cf4 \cb1 \strokec4 \
\cb3         edge \strokec9 =\strokec4  win_prob \strokec9 -\strokec4  \cf10 \strokec10 0.50\cf4 \cb1 \strokec4 \
\cb3         \cb1 \
\cb3         \cf5 \strokec5 if\cf4 \strokec4  edge \strokec9 >=\strokec4  \cf6 \strokec6 MIN_EDGE\cf4 \strokec4  \cf11 \strokec11 and\cf4 \strokec4  win_prob \strokec9 >\strokec4  \cf10 \strokec10 0.55\cf4 \strokec4 :\cb1 \
\cb3             action \strokec9 =\strokec4  \cf7 \strokec7 "BUY YES"\cf4 \strokec4  \cf5 \strokec5 if\cf4 \strokec4  win_prob \strokec9 >\strokec4  \cf10 \strokec10 0.5\cf4 \strokec4  \cf5 \strokec5 else\cf4 \strokec4  \cf7 \strokec7 "BUY NO"\cf4 \cb1 \strokec4 \
\cb3             is_signal \strokec9 =\strokec4  \cf11 \strokec11 True\cf4 \cb1 \strokec4 \
\cb3         \cf5 \strokec5 else\cf4 \strokec4 :\cb1 \
\cb3             action \strokec9 =\strokec4  \cf7 \strokec7 "SKIP"\cf4 \cb1 \strokec4 \
\cb3             is_signal \strokec9 =\strokec4  \cf11 \strokec11 False\cf4 \cb1 \strokec4 \
\cb3         \cb1 \
\cb3         \cf2 \strokec2 # Store result\cf4 \cb1 \strokec4 \
\cb3         result \strokec9 =\strokec4  \{\cb1 \
\cb3             \cf7 \strokec7 'Coin'\cf4 \strokec4 : coin_name,\cb1 \
\cb3             \cf7 \strokec7 'Price'\cf4 \strokec4 : \cf13 \strokec13 f\cf7 \strokec7 "$\cf13 \strokec13 \{\cf4 \strokec4 current_price\cf13 \strokec13 :.2f\}\cf7 \strokec7 "\cf4 \strokec4 ,\cb1 \
\cb3             \cf7 \strokec7 'Win_Prob'\cf4 \strokec4 : \cf13 \strokec13 f\cf7 \strokec7 "\cf13 \strokec13 \{\cf4 \strokec4 win_prob\cf13 \strokec13 :.0%\}\cf7 \strokec7 "\cf4 \strokec4 ,\cb1 \
\cb3             \cf7 \strokec7 'Accuracy'\cf4 \strokec4 : \cf13 \strokec13 f\cf7 \strokec7 "\cf13 \strokec13 \{\cf4 \strokec4 test_accuracy\cf13 \strokec13 :.0%\}\cf7 \strokec7 "\cf4 \strokec4 ,\cb1 \
\cb3             \cf7 \strokec7 'Edge'\cf4 \strokec4 : \cf13 \strokec13 f\cf7 \strokec7 "\cf13 \strokec13 \{\cf4 \strokec4 edge\cf13 \strokec13 :.0%\}\cf7 \strokec7 "\cf4 \strokec4 ,\cb1 \
\cb3             \cf7 \strokec7 'Spread'\cf4 \strokec4 : \cf13 \strokec13 f\cf7 \strokec7 "\cf13 \strokec13 \{\cf4 \strokec4 orderbook.get(\cf7 \strokec7 'spread'\cf4 \strokec4 , \cf10 \strokec10 0\cf4 \strokec4 )\cf13 \strokec13 :.3f\}\cf7 \strokec7 "\cf4 \strokec4 ,\cb1 \
\cb3             \cf7 \strokec7 'Imbalance'\cf4 \strokec4 : \cf13 \strokec13 f\cf7 \strokec7 "\cf13 \strokec13 \{\cf4 \strokec4 orderbook.get(\cf7 \strokec7 'imbalance'\cf4 \strokec4 , \cf10 \strokec10 0\cf4 \strokec4 )\cf13 \strokec13 :.3f\}\cf7 \strokec7 "\cf4 \strokec4 ,\cb1 \
\cb3             \cf7 \strokec7 'Change_15m'\cf4 \strokec4 : \cf13 \strokec13 f\cf7 \strokec7 "\cf13 \strokec13 \{\cf4 \strokec4 paprika_data.get(\cf7 \strokec7 'percent_change_15m'\cf4 \strokec4 , \cf10 \strokec10 0\cf4 \strokec4 )\cf13 \strokec13 :.1f\}\cf7 \strokec7 %"\cf4 \strokec4 ,\cb1 \
\cb3             \cf7 \strokec7 'Change_24h'\cf4 \strokec4 : \cf13 \strokec13 f\cf7 \strokec7 "\cf13 \strokec13 \{\cf4 \strokec4 paprika_data.get(\cf7 \strokec7 'percent_change_24h'\cf4 \strokec4 , \cf10 \strokec10 0\cf4 \strokec4 )\cf13 \strokec13 :.1f\}\cf7 \strokec7 %"\cf4 \strokec4 ,\cb1 \
\cb3             \cf7 \strokec7 'Bet_Size'\cf4 \strokec4 : \cf13 \strokec13 f\cf7 \strokec7 "$\cf13 \strokec13 \{\cf4 \strokec4 bet_amount\cf13 \strokec13 :.2f\}\cf7 \strokec7 "\cf4 \strokec4  \cf5 \strokec5 if\cf4 \strokec4  bet_amount \strokec9 >\strokec4  \cf10 \strokec10 0\cf4 \strokec4  \cf5 \strokec5 else\cf4 \strokec4  \cf7 \strokec7 "$0.00"\cf4 \strokec4 ,\cb1 \
\cb3             \cf7 \strokec7 'Action'\cf4 \strokec4 : action\cb1 \
\cb3         \}\cb1 \
\cb3         all_results.append(result)\cb1 \
\cb3         \cb1 \
\cb3         \cf5 \strokec5 if\cf4 \strokec4  is_signal:\cb1 \
\cb3             best_bets.append(result)\cb1 \
\cb3             \cb1 \
\cb3     \cf5 \strokec5 except\cf4 \strokec4  \cf14 \strokec14 Exception\cf4 \strokec4  \cf5 \strokec5 as\cf4 \strokec4  e:\cb1 \
\cb3         \cf5 \strokec5 pass\cf4 \cb1 \strokec4 \
\cb3     \cb1 \
\cb3     \cf2 \strokec2 # Update progress\cf4 \cb1 \strokec4 \
\cb3     progress_bar.progress((idx \strokec9 +\strokec4  \cf10 \strokec10 1\cf4 \strokec4 ) \strokec9 /\strokec4  \cf15 \strokec15 len\cf4 \strokec4 (\cf6 \strokec6 COINS\cf4 \strokec4 ))\cb1 \
\
\cb3 status_text.text(\cf7 \strokec7 "Done!"\cf4 \strokec4 )\cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf2 \cb3 \strokec2 # ============================================\cf4 \cb1 \strokec4 \
\cf2 \cb3 \strokec2 # \uc0\u55357 \u56522  DISPLAY RESULTS\cf4 \cb1 \strokec4 \
\cf2 \cb3 \strokec2 # ============================================\cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 st.subheader(\cf7 \strokec7 "\uc0\u55357 \u56522  Current Predictions"\cf4 \strokec4 )\cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf5 \cb3 \strokec5 if\cf4 \strokec4  all_results:\cb1 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3     df_results \strokec9 =\strokec4  pd.DataFrame(all_results)\cb1 \
\cb3     st.dataframe(df_results, \cf8 \strokec8 use_container_width\cf4 \strokec9 =\cf11 \strokec11 True\cf4 \strokec4 , \cf8 \strokec8 hide_index\cf4 \strokec9 =\cf11 \strokec11 True\cf4 \strokec4 )\cb1 \
\pard\pardeftab720\partightenfactor0
\cf5 \cb3 \strokec5 else\cf4 \strokec4 :\cb1 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3     st.warning(\cf7 \strokec7 "No results generated. Please check data sources."\cf4 \strokec4 )\cb1 \
\
\cb3 st.divider()\cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf2 \cb3 \strokec2 # --- BEST BETS SECTION ---\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 st.subheader(\cf7 \strokec7 "\uc0\u11088  Best Bets"\cf4 \strokec4 )\cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf5 \cb3 \strokec5 if\cf4 \strokec4  best_bets:\cb1 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3     total_bet \strokec9 =\strokec4  \cf10 \strokec10 0\cf4 \cb1 \strokec4 \
\cb3     \cf5 \strokec5 for\cf4 \strokec4  bet \cf5 \strokec5 in\cf4 \strokec4  best_bets:\cb1 \
\cb3         bet_size \strokec9 =\strokec4  \cf14 \strokec14 float\cf4 \strokec4 (bet[\cf7 \strokec7 'Bet_Size'\cf4 \strokec4 ].replace(\cf7 \strokec7 '$'\cf4 \strokec4 , \cf7 \strokec7 ''\cf4 \strokec4 ))\cb1 \
\cb3         total_bet \strokec9 +=\strokec4  bet_size\cb1 \
\cb3         st.success(\cf13 \strokec13 f\cf7 \strokec7 "\uc0\u9989  **\cf13 \strokec13 \{\cf4 \strokec4 bet[\cf7 \strokec7 'Coin'\cf4 \strokec4 ]\cf13 \strokec13 \}\cf7 \strokec7 **: \cf13 \strokec13 \{\cf4 \strokec4 bet[\cf7 \strokec7 'Action'\cf4 \strokec4 ]\cf13 \strokec13 \}\cf7 \strokec7 "\cf4 \strokec4 )\cb1 \
\cb3         st.caption(\cf13 \strokec13 f\cf7 \strokec7 "Win Prob: \cf13 \strokec13 \{\cf4 \strokec4 bet[\cf7 \strokec7 'Win_Prob'\cf4 \strokec4 ]\cf13 \strokec13 \}\cf7 \strokec7  | Edge: \cf13 \strokec13 \{\cf4 \strokec4 bet[\cf7 \strokec7 'Edge'\cf4 \strokec4 ]\cf13 \strokec13 \}\cf7 \strokec7  | Bet: \cf13 \strokec13 \{\cf4 \strokec4 bet[\cf7 \strokec7 'Bet_Size'\cf4 \strokec4 ]\cf13 \strokec13 \}\cf7 \strokec7  | 15m Change: \cf13 \strokec13 \{\cf4 \strokec4 bet[\cf7 \strokec7 'Change_15m'\cf4 \strokec4 ]\cf13 \strokec13 \}\cf7 \strokec7 "\cf4 \strokec4 )\cb1 \
\cb3     \cb1 \
\cb3     st.metric(\cf7 \strokec7 "Total Recommended Risk"\cf4 \strokec4 , \cf13 \strokec13 f\cf7 \strokec7 "$\cf13 \strokec13 \{\cf4 \strokec4 total_bet\cf13 \strokec13 :.2f\}\cf7 \strokec7 "\cf4 \strokec4 , \cf13 \strokec13 f\cf7 \strokec7 "\cf13 \strokec13 \{\cf4 \strokec4 total_bet\strokec9 /\cf6 \strokec6 BANKROLL\cf4 \strokec9 *\cf10 \strokec10 100\cf13 \strokec13 :.1f\}\cf7 \strokec7 % of bankroll"\cf4 \strokec4 )\cb1 \
\pard\pardeftab720\partightenfactor0
\cf5 \cb3 \strokec5 else\cf4 \strokec4 :\cb1 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3     st.info(\cf7 \strokec7 "\uc0\u9203  No bets meet the minimum edge threshold. Waiting for a better signal..."\cf4 \strokec4 )\cb1 \
\
\cb3 st.divider()\cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf2 \cb3 \strokec2 # --- DATA SOURCES ---\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf5 \cb3 \strokec5 with\cf4 \strokec4  st.expander(\cf7 \strokec7 "\uc0\u55357 \u56538  Data Sources Used"\cf4 \strokec4 ):\cb1 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3     st.markdown(\cf7 \strokec7 """\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf7 \cb3 \strokec7     \uc0\u9989  **Yahoo Finance** - OHLCV price data (1-minute candles)  \cf4 \cb1 \strokec4 \
\cf7 \cb3 \strokec7     \uc0\u9989  **Kalshi Public API** - Order book data (spread, imbalance)  \cf4 \cb1 \strokec4 \
\cf7 \cb3 \strokec7     \uc0\u9989  **CoinPaprika** - Price changes, volume, market cap, Twitter followers  \cf4 \cb1 \strokec4 \
\cf7 \cb3 \strokec7     \uc0\u10060  **Dune Analytics** - Optional on-chain metrics (disabled)\cf4 \cb1 \strokec4 \
\cf7 \cb3 \strokec7     """\cf4 \strokec4 )\cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 st.caption(\cf13 \strokec13 f\cf7 \strokec7 "\uc0\u55357 \u56622  Dashboard auto-refreshes every 60 seconds. Last update: \cf13 \strokec13 \{\cf4 \strokec4 datetime.now().strftime(\cf7 \strokec7 '%Y-%m-\cf13 \strokec13 %d\cf7 \strokec7  %H:%M:%S'\cf4 \strokec4 )\cf13 \strokec13 \}\cf7 \strokec7 "\cf4 \strokec4 )\cb1 \
}