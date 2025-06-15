import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

def detect_price_patterns(df):
    """Detect common chart patterns in price data"""
    # Make a copy to avoid modifying the original
    data = df.copy()
    
    # Calculate local minima and maxima (swing highs and lows)
    # Using 5-period window for local extrema
    data['LocalMax'] = data['Price'].rolling(window=5, center=True).apply(
        lambda x: 1 if x[2] == max(x) else 0, raw=True
    )
    data['LocalMin'] = data['Price'].rolling(window=5, center=True).apply(
        lambda x: 1 if x[2] == min(x) else 0, raw=True
    )
    
    # Identify potential patterns
    patterns = {}
    
    # Double Top pattern
    data['DoubleTop'] = 0
    for i in range(5, len(data)-5):
        if (data['LocalMax'].iloc[i] == 1 and 
            data['LocalMax'].iloc[i-5:i].sum() >= 1):
            # Check if the two tops are within 3% of each other
            prev_top_idx = data.iloc[i-5:i].loc[data['LocalMax'] == 1].index[-1]
            if abs(data['Price'].iloc[i] - data['Price'].iloc[prev_top_idx]) < 0.03 * data['Price'].iloc[prev_top_idx]:
                data.loc[data.index[i], 'DoubleTop'] = 1
                patterns[data.index[i]] = {'pattern': 'Double Top', 'signal': 'Sell'}
    
    # Double Bottom pattern
    data['DoubleBottom'] = 0
    for i in range(5, len(data)-5):
        if (data['LocalMin'].iloc[i] == 1 and 
            data['LocalMin'].iloc[i-5:i].sum() >= 1):
            # Check if the two bottoms are within 3% of each other
            prev_bottom_idx = data.iloc[i-5:i].loc[data['LocalMin'] == 1].index[-1]
            if abs(data['Price'].iloc[i] - data['Price'].iloc[prev_bottom_idx]) < 0.03 * data['Price'].iloc[prev_bottom_idx]:
                data.loc[data.index[i], 'DoubleBottom'] = 1
                patterns[data.index[i]] = {'pattern': 'Double Bottom', 'signal': 'Buy'}
    
    # Head and Shoulders pattern (simplified)
    data['HeadAndShoulders'] = 0
    for i in range(10, len(data)-5):
        if (data['LocalMax'].iloc[i-10:i-5].sum() >= 1 and 
            data['LocalMax'].iloc[i-5:i].sum() >= 1 and 
            data['LocalMax'].iloc[i] == 1):
            left_shoulder_idx = data.iloc[i-10:i-5].loc[data['LocalMax'] == 1].index[-1]
            head_idx = data.iloc[i-5:i].loc[data['LocalMax'] == 1].index[-1]
            right_shoulder_idx = i
            
            # Check if head is higher than shoulders
            if (data['Price'].iloc[head_idx] > data['Price'].iloc[left_shoulder_idx] and 
                data['Price'].iloc[head_idx] > data['Price'].iloc[right_shoulder_idx]):
                data.loc[data.index[i], 'HeadAndShoulders'] = 1
                patterns[data.index[i]] = {'pattern': 'Head and Shoulders', 'signal': 'Sell'}
    
    # Inverse Head and Shoulders pattern (simplified)
    data['InvHeadAndShoulders'] = 0
    for i in range(10, len(data)-5):
        if (data['LocalMin'].iloc[i-10:i-5].sum() >= 1 and 
            data['LocalMin'].iloc[i-5:i].sum() >= 1 and 
            data['LocalMin'].iloc[i] == 1):
            left_shoulder_idx = data.iloc[i-10:i-5].loc[data['LocalMin'] == 1].index[-1]
            head_idx = data.iloc[i-5:i].loc[data['LocalMin'] == 1].index[-1]
            right_shoulder_idx = i
            
            # Check if head is lower than shoulders
            if (data['Price'].iloc[head_idx] < data['Price'].iloc[left_shoulder_idx] and 
                data['Price'].iloc[head_idx] < data['Price'].iloc[right_shoulder_idx]):
                data.loc[data.index[i], 'InvHeadAndShoulders'] = 1
                patterns[data.index[i]] = {'pattern': 'Inverse Head and Shoulders', 'signal': 'Buy'}
    
    # Identify candlestick patterns if we have OHLC data
    if all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
        # Bullish engulfing (manual implementation)
        data['BullishEngulfing'] = 0
        for i in range(1, len(data)):
            if (data['Close'].iloc[i] > data['Open'].iloc[i] and  # Current candle is bullish
                data['Close'].iloc[i-1] < data['Open'].iloc[i-1] and  # Previous candle is bearish
                data['Close'].iloc[i] > data['Open'].iloc[i-1] and  # Current close > previous open
                data['Open'].iloc[i] < data['Close'].iloc[i-1]):  # Current open < previous close
                data.loc[data.index[i], 'BullishEngulfing'] = 1
                patterns[data.index[i]] = {'pattern': 'Bullish Engulfing', 'signal': 'Buy'}
        
        # Bearish engulfing (manual implementation)
        data['BearishEngulfing'] = 0
        for i in range(1, len(data)):
            if (data['Close'].iloc[i] < data['Open'].iloc[i] and  # Current candle is bearish
                data['Close'].iloc[i-1] > data['Open'].iloc[i-1] and  # Previous candle is bullish
                data['Close'].iloc[i] < data['Open'].iloc[i-1] and  # Current close < previous open
                data['Open'].iloc[i] > data['Close'].iloc[i-1]):  # Current open > previous close
                data.loc[data.index[i], 'BearishEngulfing'] = 1
                patterns[data.index[i]] = {'pattern': 'Bearish Engulfing', 'signal': 'Sell'}
    
    return data, patterns

def analyze_volume_profile(df):
    """Analyze volume profile to identify support/resistance levels"""
    # Make a copy to avoid modifying the original
    data = df.copy()
    
    # Create price bins
    price_min = data['Price'].min()
    price_max = data['Price'].max()
    price_range = price_max - price_min
    bin_size = price_range / 20  # 20 bins
    
    # Create bins
    bins = np.arange(price_min, price_max + bin_size, bin_size)
    labels = [(bins[i] + bins[i+1])/2 for i in range(len(bins)-1)]
    
    # Assign each price to a bin
    data['PriceBin'] = pd.cut(data['Price'], bins=bins, labels=labels)
    
    # Calculate volume profile
    volume_profile = data.groupby('PriceBin')['Volume'].sum().reset_index()
    
    # Identify high volume nodes (potential support/resistance)
    volume_threshold = volume_profile['Volume'].quantile(0.8)
    high_volume_nodes = volume_profile[volume_profile['Volume'] >= volume_threshold]
    
    # Identify current price position relative to volume nodes
    current_price = data['Price'].iloc[-1]
    
    # Convert PriceBin from categorical to float for comparison
    high_volume_nodes['PriceBin_float'] = high_volume_nodes['PriceBin'].astype(float)
    
    # Find nearest high volume nodes
    support_levels = high_volume_nodes[high_volume_nodes['PriceBin_float'] < current_price]['PriceBin_float'].tolist()
    resistance_levels = high_volume_nodes[high_volume_nodes['PriceBin_float'] > current_price]['PriceBin_float'].tolist()
    
    # Sort by distance to current price
    support_levels.sort(key=lambda x: current_price - x)
    resistance_levels.sort(key=lambda x: x - current_price)
    
    return {
        'volume_profile': volume_profile,
        'support_levels': support_levels[:3],  # Top 3 closest supports
        'resistance_levels': resistance_levels[:3],  # Top 3 closest resistances
        'high_volume_nodes': high_volume_nodes
    }

def analyze_order_flow(df):
    """Analyze order flow to identify buying/selling pressure"""
    # Make a copy to avoid modifying the original
    data = df.copy()
    
    # Calculate delta (difference between buying and selling volume)
    # If we don't have actual order flow data, we'll estimate it
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
    
    # Incorporate open interest if available
    if 'OpenInterest' in data.columns:
        data['OI_Change'] = data['OpenInterest'].diff()
        
        # Price up + OI up = strong bullish
        # Price up + OI down = weak bullish
        # Price down + OI up = strong bearish
        # Price down + OI down = weak bearish
        data['OI_Signal'] = 0
        for i in range(1, len(data)):
            price_change = data['Price_Change'].iloc[i]
            oi_change = data['OI_Change'].iloc[i]
            
            if price_change > 0 and oi_change > 0:
                data.loc[data.index[i], 'OI_Signal'] = 2  # Strong bullish
            elif price_change > 0 and oi_change < 0:
                data.loc[data.index[i], 'OI_Signal'] = 1  # Weak bullish
            elif price_change < 0 and oi_change > 0:
                data.loc[data.index[i], 'OI_Signal'] = -2  # Strong bearish
            elif price_change < 0 and oi_change < 0:
                data.loc[data.index[i], 'OI_Signal'] = -1  # Weak bearish
    
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

def backtest_strategy(df, strategy_signals, initial_capital=10000):
    """Backtest a trading strategy based on provided signals"""
    # Make a copy to avoid modifying the original
    data = df.copy()
    
    # Add strategy signals to the dataframe
    data['Signal'] = strategy_signals
    
    # Initialize portfolio metrics
    data['Position'] = 0
    data['BTC_Holdings'] = 0
    data['Cash'] = initial_capital
    data['Portfolio_Value'] = initial_capital
    
    # Execute trades based on signals
    position = 0
    btc_holdings = 0
    cash = initial_capital
    
    for i in range(1, len(data)):
        # Update position based on previous day's signal
        if data['Signal'].iloc[i-1] == 1 and position == 0:  # Buy signal
            btc_holdings = cash / data['Price'].iloc[i]
            cash = 0
            position = 1
        elif data['Signal'].iloc[i-1] == -1 and position == 1:  # Sell signal
            cash = btc_holdings * data['Price'].iloc[i]
            btc_holdings = 0
            position = 0
        
        # Update portfolio metrics
        data.loc[data.index[i], 'Position'] = position
        data.loc[data.index[i], 'BTC_Holdings'] = btc_holdings
        data.loc[data.index[i], 'Cash'] = cash
        data.loc[data.index[i], 'Portfolio_Value'] = cash + (btc_holdings * data['Price'].iloc[i])
    
    # Calculate performance metrics
    data['Returns'] = data['Portfolio_Value'].pct_change()
    data['Cumulative_Returns'] = (1 + data['Returns']).cumprod()
    
    # Calculate drawdowns
    data['Peak'] = data['Portfolio_Value'].cummax()
    data['Drawdown'] = (data['Portfolio_Value'] - data['Peak']) / data['Peak']
    
    # Calculate performance metrics
    total_return = data['Portfolio_Value'].iloc[-1] / initial_capital - 1
    max_drawdown = data['Drawdown'].min()
    win_rate = len(data[(data['Position'].diff() == -1) & (data['Cash'] > data['Cash'].shift(1))]) / max(1, len(data[data['Position'].diff() == -1]))
    
    performance = {
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'sharpe_ratio': data['Returns'].mean() / data['Returns'].std() * np.sqrt(252) if data['Returns'].std() != 0 else 0
    }
    
    return data, performance

def generate_combined_signals(df):
    """Generate trading signals based on multiple analysis techniques"""
    # Make a copy to avoid modifying the original
    data = df.copy()
    
    # Detect price patterns
    pattern_data, patterns = detect_price_patterns(data)
    
    # Analyze volume profile
    volume_analysis = analyze_volume_profile(data)
    
    # Analyze order flow
    order_flow_data = analyze_order_flow(data)
    
    # Combine all data
    for col in pattern_data.columns:
        if col not in data.columns:
            data[col] = pattern_data[col]
    
    for col in order_flow_data.columns:
        if col not in data.columns:
            data[col] = order_flow_data[col]
    
    # Generate combined signals
    data['PriceActionSignal'] = 0
    
    # Pattern-based signals
    for i in range(len(data)):
        if data.index[i] in patterns:
            if patterns[data.index[i]]['signal'] == 'Buy':
                data.loc[data.index[i], 'PriceActionSignal'] = 1
            elif patterns[data.index[i]]['signal'] == 'Sell':
                data.loc[data.index[i], 'PriceActionSignal'] = -1
    
    # Volume confirmation
    # If we're near a support level with high volume, strengthen buy signal
    current_price = data['Price'].iloc[-1]
    for support in volume_analysis['support_levels']:
        if abs(current_price - support) / current_price < 0.03:  # Within 3% of support
            if data['PriceActionSignal'].iloc[-1] >= 0:  # Not already a sell signal
                data.loc[data.index[-1], 'PriceActionSignal'] = 1
    
    # If we're near a resistance level with high volume, strengthen sell signal
    for resistance in volume_analysis['resistance_levels']:
        if abs(current_price - resistance) / current_price < 0.03:  # Within 3% of resistance
            if data['PriceActionSignal'].iloc[-1] <= 0:  # Not already a buy signal
                data.loc[data.index[-1], 'PriceActionSignal'] = -1
    
    # Order flow confirmation
    # If order flow signal agrees with price action signal, strengthen it
    for i in range(len(data)):
        if data['OrderFlowSignal'].iloc[i] == 1 and data['PriceActionSignal'].iloc[i] >= 0:
            data.loc[data.index[i], 'PriceActionSignal'] = 1
        elif data['OrderFlowSignal'].iloc[i] == -1 and data['PriceActionSignal'].iloc[i] <= 0:
            data.loc[data.index[i], 'PriceActionSignal'] = -1
    
    # Final combined signal
    data['CombinedSignal'] = data['PriceActionSignal']
    
    # If we have OI_Signal, use it to strengthen or weaken the combined signal
    if 'OI_Signal' in data.columns:
        for i in range(len(data)):
            if data['OI_Signal'].iloc[i] == 2 and data['CombinedSignal'].iloc[i] == 1:
                data.loc[data.index[i], 'CombinedSignal'] = 2  # Strong buy
            elif data['OI_Signal'].iloc[i] == -2 and data['CombinedSignal'].iloc[i] == -1:
                data.loc[data.index[i], 'CombinedSignal'] = -2  # Strong sell
    
    return data, patterns, volume_analysis

def get_current_signals(df):
    """Get current trading signals based on the latest data"""
    # Generate signals for the entire dataset
    data, patterns, volume_analysis = generate_combined_signals(df)
    
    # Get the latest signal
    latest_signal = data['CombinedSignal'].iloc[-1]
    
    # Determine action based on signal strength
    if latest_signal == 2:
        action = "Strong Buy"
    elif latest_signal == 1:
        action = "Buy"
    elif latest_signal == 0:
        action = "Hold"
    elif latest_signal == -1:
        action = "Sell"
    else:  # -2
        action = "Strong Sell"
    
    # Get the latest pattern if any
    latest_pattern = None
    if data.index[-1] in patterns:
        latest_pattern = patterns[data.index[-1]]
    
    # Get nearest support and resistance levels
    supports = volume_analysis['support_levels']
    resistances = volume_analysis['resistance_levels']
    
    current_price = df['Price'].iloc[-1]
    
    # Calculate optimal entry, exit, and stop loss
    if latest_signal > 0:  # Buy signal
        entry_price = current_price
        target_price = min(resistances) if resistances else current_price * 1.05
        stop_loss = max(supports) if supports else current_price * 0.95
    elif latest_signal < 0:  # Sell signal
        entry_price = current_price
        target_price = max(supports) if supports else current_price * 0.95
        stop_loss = min(resistances) if resistances else current_price * 1.05
    else:  # Hold
        entry_price = current_price
        target_price = current_price * 1.03
        stop_loss = current_price * 0.97
    
    # Calculate risk-reward ratio
    risk = abs(entry_price - stop_loss)
    reward = abs(target_price - entry_price)
    risk_reward = reward / risk if risk > 0 else 0
    
    # Backtest the strategy
    backtest_data, performance = backtest_strategy(df, data['CombinedSignal'])
    
    signals = {
        'action': action,
        'signal_strength': latest_signal,
        'latest_pattern': latest_pattern,
        'entry_price': entry_price,
        'target_price': target_price,
        'stop_loss': stop_loss,
        'risk_reward': risk_reward,
        'support_levels': supports,
        'resistance_levels': resistances,
        'backtest_performance': performance
    }
    
    return signals, data