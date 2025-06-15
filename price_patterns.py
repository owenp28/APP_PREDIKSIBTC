import pandas as pd
import numpy as np

def detect_patterns(df):
    """Detect common chart patterns in price data"""
    data = df.copy()
    
    # Calculate local minima and maxima (swing highs and lows)
    data['LocalMax'] = data['Price'].rolling(window=5, center=True).apply(
        lambda x: 1 if x[2] == max(x) else 0, raw=True
    )
    data['LocalMin'] = data['Price'].rolling(window=5, center=True).apply(
        lambda x: 1 if x[2] == min(x) else 0, raw=True
    )
    
    # Identify patterns
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
    
    return data, patterns

def analyze_volume_profile(df):
    """Analyze volume profile to identify support/resistance levels"""
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
        'support_levels': support_levels[:3] if support_levels else [],  # Top 3 closest supports
        'resistance_levels': resistance_levels[:3] if resistance_levels else [],  # Top 3 closest resistances
        'high_volume_nodes': high_volume_nodes
    }

def get_price_action_signals(df):
    """Generate trading signals based on price action and volume analysis"""
    # Detect patterns
    pattern_data, patterns = detect_patterns(df)
    
    # Analyze volume profile
    volume_analysis = analyze_volume_profile(df)
    
    # Generate signals
    signals = {
        'patterns': patterns,
        'latest_pattern': patterns.get(pattern_data.index[-1], None),
        'support_levels': volume_analysis['support_levels'],
        'resistance_levels': volume_analysis['resistance_levels']
    }
    
    # Determine current signal
    current_price = df['Price'].iloc[-1]
    signal = 0  # Neutral
    
    # Check if we're near support or resistance
    for support in volume_analysis['support_levels']:
        if abs(current_price - support) / current_price < 0.03:  # Within 3% of support
            signal = 1  # Buy signal
            break
    
    for resistance in volume_analysis['resistance_levels']:
        if abs(current_price - resistance) / current_price < 0.03:  # Within 3% of resistance
            signal = -1  # Sell signal
            break
    
    # Check for recent patterns
    recent_patterns = {k: v for k, v in patterns.items() if k >= df.index[-5]}
    if recent_patterns:
        latest_pattern = recent_patterns[max(recent_patterns.keys())]
        if latest_pattern['signal'] == 'Buy':
            signal = 1
        elif latest_pattern['signal'] == 'Sell':
            signal = -1
    
    # Determine action based on signal
    if signal == 1:
        action = "Buy"
    elif signal == -1:
        action = "Sell"
    else:
        action = "Hold"
    
    signals['action'] = action
    signals['signal'] = signal
    
    return signals