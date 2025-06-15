import pandas as pd
import numpy as np
from price_patterns import get_price_action_signals
from order_flow import get_order_flow_signals
from backtest import backtest_strategy

def get_combined_trading_signals(df):
    """Get combined trading signals from multiple analysis techniques"""
    # Get price action signals
    price_signals = get_price_action_signals(df)
    
    # Get order flow signals
    order_flow_signals, _ = get_order_flow_signals(df)
    
    # Combine signals
    if price_signals['signal'] == order_flow_signals['signal'] and price_signals['signal'] != 0:
        # Both agree - strong signal
        signal_strength = 2 * price_signals['signal']
        if signal_strength > 0:
            action = "Strong Buy"
        else:
            action = "Strong Sell"
    elif price_signals['signal'] != 0:
        # Price action signal
        signal_strength = price_signals['signal']
        action = price_signals['action']
    elif order_flow_signals['signal'] != 0:
        # Order flow signal
        signal_strength = order_flow_signals['signal']
        action = order_flow_signals['action']
    else:
        # No clear signal
        signal_strength = 0
        action = "Hold"
    
    # Get support and resistance levels
    support_levels = price_signals['support_levels']
    resistance_levels = price_signals['resistance_levels']
    
    # Get latest pattern
    latest_pattern = price_signals['latest_pattern']
    
    # Calculate optimal entry, exit, and stop loss
    current_price = df['Price'].iloc[-1]
    
    if signal_strength > 0:  # Buy signal
        entry_price = current_price
        target_price = min(resistance_levels) if resistance_levels else current_price * 1.05
        stop_loss = max(support_levels) * 0.98 if support_levels else current_price * 0.95
    elif signal_strength < 0:  # Sell signal
        entry_price = current_price
        target_price = max(support_levels) if support_levels else current_price * 0.95
        stop_loss = min(resistance_levels) * 1.02 if resistance_levels else current_price * 1.05
    else:  # Hold
        entry_price = current_price
        target_price = current_price * 1.03
        stop_loss = current_price * 0.97
    
    # Calculate risk-reward ratio
    risk = abs(entry_price - stop_loss)
    reward = abs(target_price - entry_price)
    risk_reward = reward / risk if risk > 0 else 0
    
    # Generate signals for backtesting
    signals = np.zeros(len(df))
    signals[-1] = 1 if signal_strength > 0 else -1 if signal_strength < 0 else 0
    
    # Backtest the strategy
    _, performance = backtest_strategy(df, signals)
    
    # Combine all signals into a dictionary
    trading_signals = {
        'action': action,
        'signal_strength': signal_strength,
        'entry_price': entry_price,
        'target_price': target_price,
        'stop_loss': stop_loss,
        'risk_reward': risk_reward,
        'latest_pattern': latest_pattern,
        'support_levels': support_levels,
        'resistance_levels': resistance_levels,
        'backtest_performance': performance,
        'order_flow': {
            'buying_pressure': order_flow_signals.get('buying_pressure', 0),
            'selling_pressure': order_flow_signals.get('selling_pressure', 0),
            'divergence': order_flow_signals.get('divergence_type', 'None')
        }
    }
    
    return trading_signals