import pandas as pd
import requests
import datetime
import numpy as np
import time

def get_current_bitcoin_price():
    """Get the current Bitcoin price from CoinMarketCap"""
    try:
        # Use CoinMarketCap API (public endpoint)
        url = "https://api.coinmarketcap.com/data-api/v3/cryptocurrency/detail/chart"
        params = {
            "id": "1",  # Bitcoin ID
            "range": "1D"  # Last 24 hours
        }
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        
        # Get the latest price point
        points = data.get("data", {}).get("points", {})
        if points:
            # Get the latest timestamp (should be the largest one)
            latest_timestamp = max(points.keys())
            latest_price = points[latest_timestamp]["v"][0]
            return latest_price
        
        # If we couldn't get the price from CoinMarketCap, try CoinGecko as backup
        return get_bitcoin_price_from_coingecko()
        
    except Exception as e:
        print(f"Error fetching current Bitcoin price from CoinMarketCap: {e}")
        # Try CoinGecko as backup
        return get_bitcoin_price_from_coingecko()

def get_bitcoin_price_from_coingecko():
    """Get the current Bitcoin price from CoinGecko as a backup"""
    try:
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {
            "ids": "bitcoin",
            "vs_currencies": "usd"
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        return data.get("bitcoin", {}).get("usd", 50000)  # Default to 50000 if not found
    
    except Exception as e:
        print(f"Error fetching current Bitcoin price from CoinGecko: {e}")
        return 50000  # Default fallback price

def load_data():
    # Use CoinGecko API to get real Bitcoin data (free, no API key required)
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    
    # Get data for last 30 days (in unix milliseconds)
    days = 30
    params = {
        "vs_currency": "usd",
        "days": days,
        "interval": "daily"
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise exception for 4XX/5XX responses
        
        data = response.json()
        
        # Extract price data (timestamp, price)
        price_data = data.get("prices", [])
        volume_data = data.get("total_volumes", [])
        
        if price_data:
            # Convert to DataFrame
            df = pd.DataFrame(price_data, columns=["timestamp", "Price"])
            
            # Add volume data
            volume_df = pd.DataFrame(volume_data, columns=["timestamp", "Volume"])
            df = pd.merge(df, volume_df, on="timestamp")
            
            # Convert timestamp to datetime
            df["Date"] = pd.to_datetime(df["timestamp"], unit="ms")
            df = df[["Date", "Price", "Volume"]]  # Keep only needed columns
            
            # Update the latest price with real-time data from CoinMarketCap
            current_price = get_current_bitcoin_price()
            df.loc[df.index[-1], "Price"] = current_price
            
            # Add technical indicators
            df = add_technical_indicators(df)
            
            return df
        else:
            raise ValueError("No price data returned from API")
            
    except Exception as e:
        print(f"Error fetching data from CoinGecko: {e}")
        
        # Fallback to simulated data if API fails
        today = datetime.datetime.now()
        dates = pd.date_range(end=today, periods=30)
        
        # Generate random but somewhat realistic prices
        np.random.seed(42)  # For reproducibility
        base_price = get_current_bitcoin_price()  # Try to get current price
        volatility = 0.03   # Daily volatility
        
        prices = [base_price]
        for i in range(1, 30):
            # Random walk with drift
            change = np.random.normal(0.001, volatility) # Slight upward drift
            prices.append(prices[-1] * (1 + change))
        
        # Reverse the list so most recent price is last
        prices.reverse()
        
        # Generate volumes
        volumes = np.random.normal(1000000000, 200000000, 30)
        
        data = pd.DataFrame({
            "Date": dates,
            "Price": prices,
            "Volume": volumes
        })
        
        # Add technical indicators
        data = add_technical_indicators(data)
        
        return data

def add_technical_indicators(df):
    """Add technical indicators to help with trading decisions"""
    # Calculate moving averages
    df['MA5'] = df['Price'].rolling(window=5).mean()
    df['MA10'] = df['Price'].rolling(window=10).mean()
    df['MA20'] = df['Price'].rolling(window=20).mean()
    
    # Calculate RSI (Relative Strength Index)
    delta = df['Price'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate MACD (Moving Average Convergence Divergence)
    df['EMA12'] = df['Price'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Price'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal']
    
    # Calculate price momentum
    df['Price_Change'] = df['Price'].pct_change() * 100
    df['Volume_Change'] = df['Volume'].pct_change() * 100
    
    # Identify potential buy/sell signals
    df['MA_Signal'] = np.where(df['MA5'] > df['MA20'], 1, -1)  # 1 for buy, -1 for sell
    df['RSI_Signal'] = np.where(df['RSI'] < 30, 1, np.where(df['RSI'] > 70, -1, 0))
    df['MACD_Signal'] = np.where(df['MACD'] > df['Signal'], 1, -1)
    
    # Combined signal (simple average of all signals)
    df['Trade_Signal'] = (df['MA_Signal'] + df['RSI_Signal'] + df['MACD_Signal']) / 3
    
    return df

def get_trading_recommendation(df):
    """Generate specific trading recommendations based on technical analysis"""
    # Get the most recent data
    recent_data = df.iloc[-5:].copy()
    
    # Calculate current market conditions
    current_price = recent_data['Price'].iloc[-1]
    rsi = recent_data['RSI'].iloc[-1]
    macd_hist = recent_data['MACD_Hist'].iloc[-1]
    
    # Calculate optimal buy and sell prices
    optimal_buy = current_price * 0.98  # Default: 2% below current
    optimal_sell = current_price * 1.02  # Default: 2% above current
    
    # Adjust based on technical indicators
    if rsi < 30:  # Oversold
        optimal_buy = current_price  # Buy now
        optimal_sell = current_price * 1.05  # Target 5% gain
    elif rsi > 70:  # Overbought
        optimal_buy = current_price * 0.95  # Wait for 5% drop
        optimal_sell = current_price  # Sell now
    
    # Adjust based on MACD
    if macd_hist > 0 and macd_hist > df['MACD_Hist'].iloc[-2]:  # Rising MACD histogram
        optimal_sell = max(optimal_sell, current_price * 1.03)  # Expect more upside
    elif macd_hist < 0 and macd_hist < df['MACD_Hist'].iloc[-2]:  # Falling MACD histogram
        optimal_buy = min(optimal_buy, current_price * 0.97)  # Expect more downside
    
    # Calculate stop loss
    stop_loss = optimal_buy * 0.95  # 5% below buy price
    
    # Calculate take profit levels
    take_profit_short = optimal_sell
    take_profit_long = optimal_sell * 1.05
    
    # Generate trading recommendation
    avg_signal = recent_data['Trade_Signal'].mean()
    
    if avg_signal > 0.5:
        action = "Strong Buy"
    elif avg_signal > 0:
        action = "Consider Buy"
    elif avg_signal > -0.5:
        action = "Hold"
    else:
        action = "Consider Sell"
    
    recommendation = {
        "action": action,
        "optimal_buy": optimal_buy,
        "optimal_sell": optimal_sell,
        "stop_loss": stop_loss,
        "take_profit_short": take_profit_short,
        "take_profit_long": take_profit_long,
        "rsi": rsi,
        "macd_histogram": macd_hist
    }
    
    return recommendation
