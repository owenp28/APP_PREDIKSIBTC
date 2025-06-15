import pandas as pd
import numpy as np

def analyze_order_flow(df):
    """Analyze order flow to identify buying/selling pressure"""
    data = df.copy()
    
    # Calculate delta (difference between buying and selling volume)
    if 'BuyVolume' not in data.columns or 'SellVolume' not in data.columns:
        # Estimate buy/sell volume based on price movement
        data['BuyVolume'] = np.where(data['Price_Change'] > 0, data['Volume'], 0)
        data['SellVolume'] = np.where(data['Price_Change'] < 0, data['Volume'], 0)
    
    data['Delta'] = data['BuyVolume'] - data['SellVolume']
    data['CumDelta'] = data['Delta'].cumsum()
    
    # Calculate delta divergence (when price moves in opposite direction to cumulative delta)
    data['DeltaDivergence'] = 0
    for i in range(5, len(data)):
        price_trend = data['Price'].iloc[i] - data['Price'].iloc[i-5]
        delta_trend = data['CumDelta'].iloc[i] - data['CumDelta'].iloc[i-5]
        
        # If price up but delta down = bearish divergence
        if price_trend > 0 and delta_trend < 0:
            data.loc[data.index[i], 'DeltaDivergence'] = -1  # Bearish
        # If price down but delta up = bullish divergence
        elif price_trend < 0 and delta_trend > 0:
            data.loc[data.index[i], 'DeltaDivergence'] = 1  # Bullish
    
    # Calculate buying/selling pressure
    data['BuyingPressure'] = data['BuyVolume'].rolling(window=5).sum()
    data['SellingPressure'] = data['SellVolume'].rolling(window=5).sum()
    data['NetPressure'] = data['BuyingPressure'] - data['SellingPressure']
    
    # Generate order flow signals
    data['OrderFlowSignal'] = 0
    data.loc[data['NetPressure'] > data['NetPressure'].quantile(0.8), 'OrderFlowSignal'] = 1  # Strong buy
    data.loc[data['NetPressure'] < data['NetPressure'].quantile(0.2), 'OrderFlowSignal'] = -1  # Strong sell
    
    # If we have delta divergence, it overrides the order flow signal
    data.loc[data['DeltaDivergence'] != 0, 'OrderFlowSignal'] = data['DeltaDivergence']
    
    return data

def get_order_flow_signals(df):
    """Get trading signals based on order flow analysis"""
    data = analyze_order_flow(df)
    
    # Get the latest signal
    latest_signal = data['OrderFlowSignal'].iloc[-1]
    
    # Calculate buying/selling pressure metrics
    buying_pressure = data['BuyingPressure'].iloc[-1]
    selling_pressure = data['SellingPressure'].iloc[-1]
    net_pressure = data['NetPressure'].iloc[-1]
    
    # Check for divergences
    has_divergence = data['DeltaDivergence'].iloc[-1] != 0
    divergence_type = "Bullish" if data['DeltaDivergence'].iloc[-1] > 0 else "Bearish" if data['DeltaDivergence'].iloc[-1] < 0 else "None"
    
    # Determine action based on signal
    if latest_signal > 0:
        action = "Buy"
    elif latest_signal < 0:
        action = "Sell"
    else:
        action = "Hold"
    
    signals = {
        'action': action,
        'signal': latest_signal,
        'buying_pressure': buying_pressure,
        'selling_pressure': selling_pressure,
        'net_pressure': net_pressure,
        'has_divergence': has_divergence,
        'divergence_type': divergence_type
    }
    
    return signals, data