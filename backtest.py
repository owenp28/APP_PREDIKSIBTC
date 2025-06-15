import pandas as pd
import numpy as np
import datetime
from data_load import get_bitcoin_ohlcv_data

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
        'sharpe_ratio': data['Returns'].mean() / data['Returns'].std() * np.sqrt(252) if data['Returns'].std() != 0 else 0,
        'last_updated': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return data, performance

def generate_combined_signals(df):
    """Generate trading signals based on multiple analysis techniques"""
    from price_patterns import get_price_action_signals
    from order_flow import get_order_flow_signals
    
    # Get price action signals
    price_signals = get_price_action_signals(df)
    
    # Get order flow signals
    order_flow_signals, _ = get_order_flow_signals(df)
    
    # Combine signals
    combined_signals = np.zeros(len(df))
    
    # Price action signals
    if price_signals['signal'] != 0:
        combined_signals[-1] = price_signals['signal']
    
    # Order flow can override if it's a strong signal
    if abs(order_flow_signals['signal']) > 0:
        combined_signals[-1] = order_flow_signals['signal']
    
    # If both agree, strengthen the signal
    if price_signals['signal'] == order_flow_signals['signal'] and price_signals['signal'] != 0:
        combined_signals[-1] = 2 * price_signals['signal']  # Strengthen the signal
    
    return combined_signals

def get_strategy_performance(df=None):
    """Get performance metrics for the trading strategy with real-time data"""
    # If no dataframe is provided, fetch fresh data
    if df is None:
        from data_load import get_bitcoin_ohlcv_data, add_technical_indicators
        df = get_bitcoin_ohlcv_data(days=30)
        df = add_technical_indicators(df)
    
    # Generate signals
    signals = generate_combined_signals(df)
    
    # Backtest the strategy
    _, performance = backtest_strategy(df, signals)
    
    return performance

def get_strategy_performance(df):
    """Get performance metrics for the trading strategy"""
    # Generate signals
    signals = generate_combined_signals(df)
    
    # Backtest the strategy
    _, performance = backtest_strategy(df, signals)
    
    return performance
