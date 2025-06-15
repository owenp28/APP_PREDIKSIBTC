import pandas as pd
import numpy as np
import datetime
import time
import threading
from data_load import get_current_bitcoin_price, get_bitcoin_ohlcv_data

# Global variables to store data that will be updated periodically
portfolio_data = {
    'allocation': None,
    'metrics': None,
    'last_updated': None
}

# Lock for thread safety
data_lock = threading.Lock()

def get_real_time_portfolio_data():
    """Get portfolio allocation and performance data using real-time Bitcoin data"""
    # Define portfolio assets
    portfolio = {
        'VTI': 'Vanguard Total Stock Market ETF',
        'VNQ': 'Vanguard Real Estate ETF',
        'VXUS': 'Vanguard Total International Stock ETF',
        'BND': 'Vanguard Total Bond Market ETF',
        'BTC-USD': 'Bitcoin'
    }
    
    try:
        # Get Bitcoin data from our own data loader
        btc_data = get_bitcoin_ohlcv_data(days=30)
        
        # Calculate Bitcoin returns
        btc_data['Returns'] = btc_data['Price'].pct_change()
        
        # Calculate metrics for Bitcoin
        btc_annual_return = btc_data['Returns'].mean() * 252
        btc_std_dev = btc_data['Returns'].std() * np.sqrt(252)
        
        # Calculate Bitcoin drawdown
        btc_cum_returns = (1 + btc_data['Returns'].fillna(0)).cumprod()
        btc_running_max = btc_cum_returns.cummax()
        btc_drawdown = (btc_cum_returns / btc_running_max) - 1
        btc_max_drawdown = btc_drawdown.min()
        
        # Create metrics for all assets (using simulated data for ETFs)
        metrics_df = pd.DataFrame(index=['VTI', 'VNQ', 'VXUS', 'BND', 'BTC-USD'])
        
        # Simulated returns for ETFs (more realistic values)
        metrics_df.loc['VTI', 'Annualized Return'] = 0.09 + np.random.normal(0, 0.01)
        metrics_df.loc['VNQ', 'Annualized Return'] = 0.07 + np.random.normal(0, 0.01)
        metrics_df.loc['VXUS', 'Annualized Return'] = 0.06 + np.random.normal(0, 0.01)
        metrics_df.loc['BND', 'Annualized Return'] = 0.03 + np.random.normal(0, 0.005)
        metrics_df.loc['BTC-USD', 'Annualized Return'] = btc_annual_return
        
        # Simulated risk metrics for ETFs
        metrics_df.loc['VTI', 'Standard Deviation'] = 0.15 + np.random.normal(0, 0.01)
        metrics_df.loc['VNQ', 'Standard Deviation'] = 0.18 + np.random.normal(0, 0.01)
        metrics_df.loc['VXUS', 'Standard Deviation'] = 0.16 + np.random.normal(0, 0.01)
        metrics_df.loc['BND', 'Standard Deviation'] = 0.04 + np.random.normal(0, 0.005)
        metrics_df.loc['BTC-USD', 'Standard Deviation'] = btc_std_dev
        
        # Simulated drawdowns
        metrics_df.loc['VTI', 'Maximum Drawdown'] = -0.20 + np.random.normal(0, 0.02)
        metrics_df.loc['VNQ', 'Maximum Drawdown'] = -0.25 + np.random.normal(0, 0.02)
        metrics_df.loc['VXUS', 'Maximum Drawdown'] = -0.22 + np.random.normal(0, 0.02)
        metrics_df.loc['BND', 'Maximum Drawdown'] = -0.08 + np.random.normal(0, 0.01)
        metrics_df.loc['BTC-USD', 'Maximum Drawdown'] = btc_max_drawdown
        
        # Benchmark relative (using S&P 500 as benchmark - simulated)
        benchmark_return = 0.08  # Simulated S&P 500 return
        
        # Add benchmark relative to metrics
        for ticker in metrics_df.index:
            metrics_df.loc[ticker, 'Benchmark Relative'] = metrics_df.loc[ticker, 'Annualized Return'] - benchmark_return
        
        # Portfolio allocation (dynamic based on performance)
        # Better performers get slightly higher allocation
        performance_scores = (metrics_df['Annualized Return'] / metrics_df['Standard Deviation']).rank()
        total_score = performance_scores.sum()
        allocations = (performance_scores / total_score * 100).round(1)
        
        allocation = pd.DataFrame({
            'Ticker': list(portfolio.keys()),
            'Name': list(portfolio.values()),
            'Allocation': allocations.values
        })
        
        return allocation, metrics_df
        
    except Exception as e:
        print(f"Error fetching portfolio data: {e}")
        
        # Create dummy allocation
        allocation = pd.DataFrame({
            'Ticker': list(portfolio.keys()),
            'Name': list(portfolio.values()),
            'Allocation': [20.0] * len(portfolio)
        })
        
        # Create dummy metrics
        metrics_df = pd.DataFrame(index=portfolio.keys())
        metrics_df['Annualized Return'] = [0.15, 0.08, 0.10, 0.04, 0.25]
        metrics_df['Standard Deviation'] = [0.18, 0.22, 0.20, 0.05, 0.65]
        metrics_df['Maximum Drawdown'] = [-0.25, -0.30, -0.28, -0.10, -0.55]
        metrics_df['Benchmark Relative'] = [0.03, -0.04, -0.02, -0.08, 0.13]
        
        return allocation, metrics_df

def update_portfolio_data_thread():
    """Background thread to update portfolio data periodically"""
    while True:
        try:
            # Get fresh data
            allocation, metrics = get_real_time_portfolio_data()
            
            # Update global data with lock to ensure thread safety
            with data_lock:
                portfolio_data['allocation'] = allocation
                portfolio_data['metrics'] = metrics
                portfolio_data['last_updated'] = datetime.datetime.now()
                
            # Sleep for 5 minutes before updating again
            time.sleep(300)
        except Exception as e:
            print(f"Error in update thread: {e}")
            time.sleep(60)  # Shorter sleep on error

def get_portfolio_data():
    """Get the latest portfolio data (used by the app)"""
    with data_lock:
        if portfolio_data['allocation'] is None or portfolio_data['metrics'] is None:
            # Initial load
            allocation, metrics = get_real_time_portfolio_data()
            portfolio_data['allocation'] = allocation
            portfolio_data['metrics'] = metrics
            portfolio_data['last_updated'] = datetime.datetime.now()
            
            # Start background update thread if not already running
            if not hasattr(get_portfolio_data, 'thread_started'):
                update_thread = threading.Thread(target=update_portfolio_data_thread, daemon=True)
                update_thread.start()
                get_portfolio_data.thread_started = True
        
        return portfolio_data['allocation'], portfolio_data['metrics']
