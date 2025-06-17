import pandas as pd
import requests
import datetime
import numpy as np
import time

def get_current_bitcoin_price():
    """Get the current Bitcoin price from Indodax BTCIDR market"""
    try:
        # Use Indodax API for BTCIDR
        url = "https://indodax.com/api/ticker/btcidr"
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        
        # Get the latest price (in IDR)
        latest_price_idr = float(data.get("ticker", {}).get("last", 0))
        
        # Convert IDR to USD (approximate conversion - in a real app you'd use a currency API)
        # Using a fixed rate for simplicity - you might want to use a real-time exchange rate
        idr_to_usd_rate = 0.000064  # Approximate rate, update as needed
        latest_price_usd = latest_price_idr * idr_to_usd_rate
        
        return latest_price_usd
        
    except Exception as e:
        print(f"Error fetching current Bitcoin price from Indodax: {e}")
        # Fall back to CoinGecko as backup
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

def get_bitcoin_ohlcv_data(days=30, interval='1d'):
    """Get Bitcoin OHLCV (Open, High, Low, Close, Volume) data from Indodax BTCIDR market"""
    try:
        # Use Indodax API for BTCIDR trade history
        url = "https://indodax.com/api/btcidr/trades"
        
        response = requests.get(url)
        response.raise_for_status()
        
        # trades_data = response.json()  # Not used
        
        # Get chart data from Indodax
        chart_url = "https://indodax.com/tradingview/history"
        params = {
            "symbol": "BTCIDR",
            "resolution": "D",  # Daily resolution
            "from": int(time.time()) - (days * 86400),  # days ago in seconds
            "to": int(time.time())  # current time
        }
        chart_response = requests.get(chart_url, params=params)
        chart_response.raise_for_status()
        
        chart_data = chart_response.json()
        
        # Create DataFrame from chart data
        df = pd.DataFrame({
            'Date': pd.to_datetime(chart_data['t'], unit='s'),
            'Open': chart_data['o'],
            'High': chart_data['h'],
            'Low': chart_data['l'],
            'Close': chart_data['c'],
            'Volume': chart_data['v']
        })
        
        # Add Price column (same as Close for consistency with previous code)
        df['Price'] = df['Close']
        
        # Calculate buy/sell volume based on price movement
        df['BuyVolume'] = np.where(df['Close'] >= df['Open'], df['Volume'], 0)
        df['SellVolume'] = np.where(df['Close'] < df['Open'], df['Volume'], 0)
        
        # Convert IDR to USD for consistency with the rest of the app
        # Using a fixed rate for simplicity - you might want to use a real-time exchange rate
        idr_to_usd_rate = 0.000064  # Approximate rate, update as needed
        
        for col in ['Open', 'High', 'Low', 'Close', 'Price']:
            df[col] = df[col] * idr_to_usd_rate
        
        # Add placeholder for open interest
        df['OpenInterest'] = 0
        
        return df
    
    except Exception as e:
        print(f"Error fetching OHLCV data from Indodax: {e}")
        # Fall back to CoinGecko
        return get_bitcoin_ohlcv_from_coingecko(days)

def get_bitcoin_ohlcv_from_coingecko(days=30):
    """Get Bitcoin OHLCV data from CoinGecko as a backup"""
    try:
        url = "https://api.coingecko.com/api/v3/coins/bitcoin/ohlc"
        params = {
            "vs_currency": "usd",
            "days": days
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        # Convert to DataFrame
        df = pd.DataFrame(data, columns=['timestamp', 'Open', 'High', 'Low', 'Close'])
        
        # Convert timestamp to datetime
        df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.drop('timestamp', axis=1)
        
        # Add Price column (same as Close for consistency)
        df['Price'] = df['Close']
        
        # Get volume data separately
        volume_url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
        volume_params = {
            "vs_currency": "usd",
            "days": days,
            "interval": "daily"
        }
        
        volume_response = requests.get(volume_url, params=volume_params)
        volume_response.raise_for_status()
        
        volume_data = volume_response.json()
        
        # Extract volume data
        volumes = volume_data.get("total_volumes", [])
        volume_df = pd.DataFrame(volumes, columns=["timestamp", "Volume"])
        volume_df['timestamp'] = pd.to_datetime(volume_df['timestamp'], unit='ms')
        
        # Merge with main dataframe
        df = pd.merge_asof(df, volume_df, left_on='Date', right_on='timestamp', direction='nearest')
        df = df.drop('timestamp', axis=1)
        
        # Estimate buy/sell volume based on price movement
        df['BuyVolume'] = np.where(df['Close'] >= df['Open'], df['Volume'], 0)
        df['SellVolume'] = np.where(df['Close'] < df['Open'], df['Volume'], 0)
        
        # Add placeholder for open interest
        df['OpenInterest'] = 0
        
        return df
    
    except Exception as e:
        print(f"Error fetching OHLCV data from CoinGecko: {e}")
        return generate_simulated_ohlcv_data(days)

def generate_simulated_ohlcv_data(days=30):
    """Generate simulated OHLCV data as a last resort"""
    today = datetime.datetime.now()
    dates = pd.date_range(end=today, periods=days)
    
    # Generate random but somewhat realistic prices
    np.random.seed(42)  # For reproducibility
    base_price = get_current_bitcoin_price()  # Try to get current price
    volatility = 0.03   # Daily volatility
    
    # Generate OHLCV data
    data = []
    prev_close = base_price
    
    for i in range(days):
        # Random walk with drift
        daily_return = np.random.normal(0.001, volatility)
        close = prev_close * (1 + daily_return)
        
        # Generate intraday range
        high = close * (1 + abs(np.random.normal(0, volatility/2)))
        low = close * (1 - abs(np.random.normal(0, volatility/2)))
        
        # Ensure high is the highest and low is the lowest
        if high < close:
            high = close * 1.01
        if low > close:
            low = close * 0.99
        
        # Generate open price
        if i == 0:
            open_price = prev_close
        else:
            open_price = prev_close * (1 + np.random.normal(0, volatility/3))
        
        # Ensure open is within high-low range
        open_price = max(min(open_price, high), low)
        
        # Generate volume
        volume = np.random.normal(1000000000, 200000000)
        
        data.append({
            'Date': dates[i],
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close,
            'Price': close,
            'Volume': volume,
            'BuyVolume': volume if close >= open_price else 0,
            'SellVolume': volume if close < open_price else 0,
            'OpenInterest': 0
        })
        
        prev_close = close
    
    df = pd.DataFrame(data)
    return df

def load_data():
    """Load Bitcoin price data with enhanced features from Indodax BTCIDR market"""
    # Get OHLCV data from Indodax
    df = get_bitcoin_ohlcv_data(days=30)
    
    # Add technical indicators
    df = add_technical_indicators(df)
    
    return df

def add_technical_indicators(df):
    """Add technical indicators to help with trading decisions"""
    # Make a copy to avoid modifying the original
    data = df.copy()
    
    # Calculate moving averages
    data['MA5'] = data['Price'].rolling(window=5).mean()
    data['MA10'] = data['Price'].rolling(window=10).mean()
    data['MA20'] = data['Price'].rolling(window=20).mean()
    
    # Calculate RSI (Relative Strength Index)
    delta = data['Price'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate MACD (Moving Average Convergence Divergence)
    data['EMA12'] = data['Price'].ewm(span=12, adjust=False).mean()
    data['EMA26'] = data['Price'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['MACD_Hist'] = data['MACD'] - data['Signal']
    
    # Calculate price momentum
    data['Price_Change'] = data['Price'].pct_change() * 100
    data['Volume_Change'] = data['Volume'].pct_change() * 100
    
    # Identify potential buy/sell signals
    data['MA_Signal'] = np.where(data['MA5'] > data['MA20'], 1, -1)  # 1 for buy, -1 for sell
    data['RSI_Signal'] = np.where(data['RSI'] < 30, 1, np.where(data['RSI'] > 70, -1, 0))
    data['MACD_Signal'] = np.where(data['MACD'] > data['Signal'], 1, -1)
    
    # Combined signal (simple average of all signals)
    data['Trade_Signal'] = (data['MA_Signal'] + data['RSI_Signal'] + data['MACD_Signal']) / 3
    
    return data

def get_trading_recommendation(df):
    """Generate specific trading recommendations based on technical analysis and price action"""
    from advanced_analysis import get_current_signals
    
    # Get advanced signals
    signals, _ = get_current_signals(df)
    
    # Get the most recent data
    recent_data = df.iloc[-5:].copy()
    
    # Calculate current market conditions
    current_price = recent_data['Price'].iloc[-1]
    rsi = recent_data['RSI'].iloc[-1]
    macd_hist = recent_data['MACD_Hist'].iloc[-1]
    
    # Use signals from advanced analysis
    action = signals['action']
    optimal_buy = signals['entry_price'] if signals['signal_strength'] >= 0 else signals['target_price']
    optimal_sell = signals['entry_price'] if signals['signal_strength'] <= 0 else signals['target_price']
    stop_loss = signals['stop_loss']
    
    # Calculate take profit levels
    take_profit_short = optimal_sell if signals['signal_strength'] >= 0 else optimal_buy
    take_profit_long = take_profit_short * 1.05 if signals['signal_strength'] >= 0 else take_profit_short * 0.95
    
    # Add timestamp for when recommendation was generated
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Create recommendation dictionary
    recommendation = {
        "action": action,
        "optimal_buy": optimal_buy,
        "optimal_sell": optimal_sell,
        "stop_loss": stop_loss,
        "take_profit_short": take_profit_short,
        "take_profit_long": take_profit_long,
        "rsi": rsi,
        "macd_histogram": macd_hist,
        "pattern": signals['latest_pattern']['pattern'] if signals['latest_pattern'] else None,
        "risk_reward": signals['risk_reward'],
        "support_levels": signals['support_levels'],
        "resistance_levels": signals['resistance_levels'],
        "backtest_performance": signals['backtest_performance'],
        "last_updated": timestamp
    }
    
    return recommendation
