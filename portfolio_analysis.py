import pandas as pd
import numpy as np
import datetime
import requests

def get_portfolio_data():
    """Get portfolio allocation and performance data"""
    # Define portfolio assets
    portfolio = {
        'VTI': 'Vanguard Total Stock Market ETF',
        'VNQ': 'Vanguard Real Estate ETF',
        'VXUS': 'Vanguard Total International Stock ETF',
        'BND': 'Vanguard Total Bond Market ETF',
        'BTC-USD': 'Bitcoin'
    }
    
    try:
        # Get current Bitcoin price
        btc_price = get_bitcoin_price()
        
        # Create metrics for all assets (using simulated data)
        metrics_df = pd.DataFrame(index=['VTI', 'VNQ', 'VXUS', 'BND', 'BTC-USD'])
        
        # Simulated returns
        metrics_df.loc['VTI', 'Annualized Return'] = 0.09 + np.random.normal(0, 0.01)
        metrics_df.loc['VNQ', 'Annualized Return'] = 0.07 + np.random.normal(0, 0.01)
        metrics_df.loc['VXUS', 'Annualized Return'] = 0.06 + np.random.normal(0, 0.01)
        metrics_df.loc['BND', 'Annualized Return'] = 0.03 + np.random.normal(0, 0.005)
        metrics_df.loc['BTC-USD', 'Annualized Return'] = 0.25 + np.random.normal(0, 0.05)
        
        # Simulated risk metrics
        metrics_df.loc['VTI', 'Standard Deviation'] = 0.15 + np.random.normal(0, 0.01)
        metrics_df.loc['VNQ', 'Standard Deviation'] = 0.18 + np.random.normal(0, 0.01)
        metrics_df.loc['VXUS', 'Standard Deviation'] = 0.16 + np.random.normal(0, 0.01)
        metrics_df.loc['BND', 'Standard Deviation'] = 0.04 + np.random.normal(0, 0.005)
        metrics_df.loc['BTC-USD', 'Standard Deviation'] = 0.65 + np.random.normal(0, 0.05)
        
        # Simulated drawdowns
        metrics_df.loc['VTI', 'Maximum Drawdown'] = -0.20 + np.random.normal(0, 0.02)
        metrics_df.loc['VNQ', 'Maximum Drawdown'] = -0.25 + np.random.normal(0, 0.02)
        metrics_df.loc['VXUS', 'Maximum Drawdown'] = -0.22 + np.random.normal(0, 0.02)
        metrics_df.loc['BND', 'Maximum Drawdown'] = -0.08 + np.random.normal(0, 0.01)
        metrics_df.loc['BTC-USD', 'Maximum Drawdown'] = -0.55 + np.random.normal(0, 0.05)
        
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

def get_bitcoin_price():
    """Get current Bitcoin price from CoinGecko API"""
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
        print(f"Error fetching Bitcoin price: {e}")
        return 50000  # Default fallback price
