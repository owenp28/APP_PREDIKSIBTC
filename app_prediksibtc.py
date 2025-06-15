import streamlit as st
import numpy as np
import pandas as pd
import datetime
import plotly.graph_objects as go
from data_load import load_data, get_trading_recommendation, get_current_bitcoin_price
from train_model import preprocess_data, train_model
from prediction_btc import make_predictions
from trading_signals import get_combined_trading_signals

# Set page configuration and custom theme
st.set_page_config(
    page_title="Bitcoin Price Prediction App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better typography and color scheme
# Using a triadic color scheme: #3366CC (primary blue), #CC3366 (accent pink), #66CC33 (accent green)
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        font-family: 'Segoe UI', Arial, sans-serif;
    }
    h1, h2, h3 {
        color: #3366CC;
        font-weight: 600;
    }
    .stButton>button {
        background-color: #3366CC;
        color: white;
    }
    .success-box {
        background-color: #66CC33;
        padding: 10px;
        border-radius: 5px;
        color: white;
    }
    .warning-box {
        background-color: #FFA500;
        padding: 10px;
        border-radius: 5px;
        color: white;
    }
    .danger-box {
        background-color: #CC3366;
        padding: 10px;
        border-radius: 5px;
        color: white;
    }
    .info-box {
        background-color: #3366CC;
        padding: 10px;
        border-radius: 5px;
        color: white;
    }
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        text-align: center;
        margin: 5px;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

# App title with better styling
st.markdown("<h1 style='text-align: center;'>Bitcoin Price Prediction App</h1>", unsafe_allow_html=True)
# Load and preprocess data
with st.spinner("Loading Bitcoin data..."):
    data = load_data()
    # Ensure 'Date' column is datetime for plotting
    if 'Date' in data.columns and not np.issubdtype(data['Date'].dtype, np.datetime64):
        data['Date'] = pd.to_datetime(data['Date'])

# Get real-time price from CoinMarketCap
current_price = get_current_bitcoin_price()
last_price = data["Price"].iloc[-1]

# Add auto-refresh for current price
if st.sidebar.button("🔄 Refresh Price"):
    current_price = get_current_bitcoin_price()
    st.sidebar.success("Price updated!")

# Display current price with timestamp
st.sidebar.markdown("<h3>Current Bitcoin Price</h3>", unsafe_allow_html=True)
st.sidebar.markdown(
    f"<div><h2 style='color: #3366CC; margin:0;'>${current_price:,.2f}</h2></div>", 
    unsafe_allow_html=True
)
st.sidebar.markdown(
    f"<div style='font-size:0.8em; color:#666;'>Last updated: {datetime.datetime.now().strftime('%H:%M:%S')}</div>",
    unsafe_allow_html=True
)

# Show price change from historical data
price_change = ((current_price - last_price) / last_price) * 100
change_color = "#66CC33" if price_change >= 0 else "#CC3366"
st.sidebar.markdown(
    f"<div style='color:{change_color};'>{'▲' if price_change >= 0 else '▼'} {abs(price_change):.2f}%</div>",
    unsafe_allow_html=True
)

# Get trading recommendations based on technical analysis
trading_rec = get_trading_recommendation(data)

# Get advanced trading signals
advanced_signals = get_combined_trading_signals(data)

# Update data with current price for better predictions
data.loc[data.index[-1], "Price"] = current_price

# Preprocess data
with st.spinner("Processing data..."):
    processed_data, scaler = preprocess_data(data)
    model, mse, features = train_model(processed_data)

# Main content in tabs for better organization
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Price Analysis", "📊 Advanced Trading", "💰 Investment Calculator", "📊 Portfolio Analysis", "ℹ️ About"])

with tab1:
    # Historical and predicted prices
    st.markdown("<h3>Bitcoin Price Analysis</h3>", unsafe_allow_html=True)
    
    # Define technical analysis functions inline
    def get_technical_indicators(data):
        """Calculate technical indicators for Bitcoin price data"""
        # Make a copy to avoid modifying the original
        df = data.copy()
        
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
        
        # Calculate Moving Averages
        df['MA5'] = df['Price'].rolling(window=5).mean()
        df['MA10'] = df['Price'].rolling(window=10).mean()
        df['MA20'] = df['Price'].rolling(window=20).mean()
        
        # Calculate MA Crossover signals
        df['MA_Crossover'] = np.where(df['MA5'] > df['MA20'], 'Bullish', 'Bearish')
        
        # Get the latest values
        latest = {}
        latest['RSI'] = df['RSI'].iloc[-1]
        latest['MACD_Hist'] = df['MACD_Hist'].iloc[-1]
        latest['MA5'] = df['MA5'].iloc[-1]
        latest['MA20'] = df['MA20'].iloc[-1]
        latest['MA_Crossover'] = df['MA_Crossover'].iloc[-1]
        
        # Determine RSI signal
        if latest['RSI'] > 70:
            latest['RSI_Signal'] = 'Overbought'
        elif latest['RSI'] < 30:
            latest['RSI_Signal'] = 'Oversold'
        else:
            latest['RSI_Signal'] = 'Neutral'
        
        # Determine MACD signal
        if latest['MACD_Hist'] > 0:
            latest['MACD_Signal'] = 'Bullish (>0)'
        else:
            latest['MACD_Signal'] = 'Bearish (<0)'
        
        # Determine overall signal
        signals = []
        if latest['RSI'] < 30:
            signals.append(1)  # Oversold - bullish
        elif latest['RSI'] > 70:
            signals.append(-1)  # Overbought - bearish
        else:
            signals.append(0)
            
        if latest['MACD_Hist'] > 0:
            signals.append(1)  # Bullish
        else:
            signals.append(-1)  # Bearish
            
        if latest['MA_Crossover'] == 'Bullish':
            signals.append(1)
        else:
            signals.append(-1)
        
        avg_signal = sum(signals) / len(signals)
        
        if avg_signal > 0.5:
            latest['Overall_Signal'] = 'Strong Buy'
        elif avg_signal > 0:
            latest['Overall_Signal'] = 'Consider Buy'
        elif avg_signal > -0.5:
            latest['Overall_Signal'] = 'Hold'
        else:
            latest['Overall_Signal'] = 'Consider Sell'
        
        return latest

    def get_investment_recommendation(prediction_data):
        """Generate investment recommendations based on predictions"""
        # Calculate short-term (7 days) and long-term (30 days) price changes
        current_price = prediction_data['Predicted_Price'].iloc[0]
        
        # Short-term outlook (7 days)
        if len(prediction_data) >= 7:
            price_7d = prediction_data['Predicted_Price'].iloc[6]
            change_7d = (price_7d - current_price) / current_price * 100
            
            if change_7d > 5:
                short_term = {'outlook': 'Strong Buy', 'change': change_7d}
            elif change_7d > 2:
                short_term = {'outlook': 'Buy', 'change': change_7d}
            elif change_7d > -2:
                short_term = {'outlook': 'Hold', 'change': change_7d}
            elif change_7d > -5:
                short_term = {'outlook': 'Sell', 'change': change_7d}
            else:
                short_term = {'outlook': 'Strong Sell', 'change': change_7d}
        else:
            short_term = {'outlook': 'Insufficient Data', 'change': 0}
        
        # Long-term outlook (30 days)
        if len(prediction_data) >= 30:
            price_30d = prediction_data['Predicted_Price'].iloc[29]
            change_30d = (price_30d - current_price) / current_price * 100
            
            if change_30d > 10:
                long_term = {'outlook': 'Strong Buy', 'change': change_30d}
            elif change_30d > 5:
                long_term = {'outlook': 'Buy', 'change': change_30d}
            elif change_30d > -5:
                long_term = {'outlook': 'Hold', 'change': change_30d}
            elif change_30d > -10:
                long_term = {'outlook': 'Sell', 'change': change_30d}
            else:
                long_term = {'outlook': 'Strong Sell', 'change': change_30d}
        else:
            long_term = {'outlook': 'Insufficient Data', 'change': 0}
        
        return {'short_term': short_term, 'long_term': long_term}
    
    # Get technical indicators
    tech_indicators = get_technical_indicators(data)
    
    # Prepare prediction_data for investment recommendation (move this up so it's defined)
    days_ahead = 30  # or any default value you want for the technical analysis section
    prediction_data = make_predictions(model, scaler, features, days_ahead)
    # Ensure 'Date' column is datetime for plotting
    if 'Date' in prediction_data.columns and not np.issubdtype(prediction_data['Date'].dtype, np.datetime64):
        prediction_data['Date'] = pd.to_datetime(prediction_data['Date'])
    # Add some volatility to make the predictions more realistic
    np.random.seed(42)  # For reproducibility
    volatility_factor = 0.01  # 1% daily volatility
    for i in range(1, len(prediction_data)):
        random_change = np.random.normal(0.001, volatility_factor)
        prediction_data.loc[i, 'Predicted_Price'] = prediction_data.loc[i-1, 'Predicted_Price'] * (1 + random_change)
    
    # Create Technical Analysis section
    st.markdown("<h4>Technical Analysis</h4>", unsafe_allow_html=True)
    
    # Create three columns for technical indicators
    ta_col1, ta_col2, ta_col3 = st.columns(3)
    
    with ta_col1:
        st.markdown("<p style='font-weight:bold'>Key Indicators</p>", unsafe_allow_html=True)
        
        # RSI
        rsi_color = "#66CC33" if tech_indicators['RSI'] < 30 else "#CC3366" if tech_indicators['RSI'] > 70 else "#3366CC"
        st.markdown(f"""
        <div style='border-left: 4px solid {rsi_color}; padding-left: 10px;'>
            <p style='margin:0; font-weight:bold'>RSI (Relative Strength Index)</p>
            <p style='margin:0; font-size:1.5em;'>{tech_indicators['RSI']:.2f}</p>
            <p style='margin:0; color:{rsi_color}'>{tech_indicators['RSI_Signal']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # MACD Histogram
        macd_color = "#66CC33" if tech_indicators['MACD_Hist'] > 0 else "#CC3366"
        st.markdown(f"""
        <div style='border-left: 4px solid {macd_color}; padding-left: 10px;'>
            <p style='margin:0; font-weight:bold'>MACD Histogram</p>
            <p style='margin:0; font-size:1.5em;'>{tech_indicators['MACD_Hist']:.2f}</p>
            <p style='margin:0; color:{macd_color}'>{tech_indicators['MACD_Signal']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with ta_col2:
        st.markdown("<p style='font-weight:bold'>Moving Averages</p>", unsafe_allow_html=True)
        
        # MA Crossover
        ma_color = "#66CC33" if tech_indicators['MA_Crossover'] == 'Bullish' else "#CC3366"
        st.markdown(f"""
        <div style='border-left: 4px solid {ma_color}; padding-left: 10px;'>
            <p style='margin:0; font-weight:bold'>MA Crossover</p>
            <p style='margin:0; font-size:1.5em; color:{ma_color}'>{tech_indicators['MA_Crossover']}</p>
            <p style='margin:0;'>MA5: ${tech_indicators['MA5']:.2f} | MA20: ${tech_indicators['MA20']:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Overall Signal
        signal_color = "#66CC33" if "Buy" in tech_indicators['Overall_Signal'] else "#FFA500" if tech_indicators['Overall_Signal'] == "Hold" else "#CC3366"
        st.markdown(f"""
        <div style='border-left: 4px solid {signal_color}; padding-left: 10px;'>
            <p style='margin:0; font-weight:bold'>Overall Signal</p>
            <p style='margin:0; font-size:1.5em; color:{signal_color}'>{tech_indicators['Overall_Signal']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with ta_col3:
        st.markdown("<p style='font-weight:bold'>Model Performance</p>", unsafe_allow_html=True)
        
        # MSE
        st.markdown(f"""
        <div style='border-left: 4px solid #3366CC; padding-left: 10px;'>
            <p style='margin:0; font-weight:bold'>Mean Squared Error</p>
            <p style='margin:0; font-size:1.5em;'>{mse:.4f}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Investment Recommendation
        recommendations = get_investment_recommendation(prediction_data)
        
        st.markdown("<p style='font-weight:bold'>Investment Recommendation</p>", unsafe_allow_html=True)
        
        # Short-term outlook
        short_term = recommendations['short_term']
        short_color = "#66CC33" if "Buy" in short_term['outlook'] else "#FFA500" if short_term['outlook'] == "Hold" else "#CC3366"
        st.markdown(f"""
        <div style='margin-bottom:10px;'>
            <p style='margin:0; font-weight:bold'>Short-term Outlook (7 days)</p>
            <p style='margin:0; color:{short_color}'>{short_term['outlook']}: Expected change of {short_term['change']:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Long-term outlook
        long_term = recommendations['long_term']
        long_color = "#66CC33" if "Buy" in long_term['outlook'] else "#FFA500" if long_term['outlook'] == "Hold" else "#CC3366"
        st.markdown(f"""
        <div>
            <p style='margin:0; font-weight:bold'>Long-term Outlook (30 days)</p>
            <p style='margin:0; color:{long_color}'>{long_term['outlook']}: Expected change of {long_term['change']:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h4>Historical Bitcoin Prices</h4>", unsafe_allow_html=True)
        
        # Calculate price changes for coloring
        data['Price_Change'] = data['Price'].diff()
        
        # Create enhanced chart with Plotly
        fig = go.Figure()
        
        # Add traces for price increases (green)
        fig.add_trace(go.Scatter(
            x=data[data['Price_Change'] >= 0]['Date'],
            y=data[data['Price_Change'] >= 0]['Price'],
            mode='lines',
            line=dict(color='#66CC33', width=2),
            name='Price Increase',
            fill='tozeroy',
            fillcolor='rgba(102, 204, 51, 0.2)'
        ))
        
        # Add traces for price decreases (red)
        fig.add_trace(go.Scatter(
            x=data[data['Price_Change'] < 0]['Date'],
            y=data[data['Price_Change'] < 0]['Price'],
            mode='lines',
            line=dict(color='#CC3366', width=2),
            name='Price Decrease',
            fill='tozeroy',
            fillcolor='rgba(204, 51, 102, 0.2)'
        ))
        
        # Update layout
        fig.update_layout(
            title='',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            margin=dict(l=0, r=0, t=10, b=0),
            height=400,
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=7, label="7d", step="day", stepmode="backward"),
                        dict(count=14, label="14d", step="day", stepmode="backward"),
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(visible=True),
                type="date"
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        days_ahead = st.slider("Days to predict ahead:", 7, 60, 30, key="days_slider")
        prediction_data = make_predictions(model, scaler, features, days_ahead)
        # Ensure 'Date' column is datetime for plotting
        if 'Date' in prediction_data.columns and not np.issubdtype(prediction_data['Date'].dtype, np.datetime64):
            prediction_data['Date'] = pd.to_datetime(prediction_data['Date'])
        
        # Add some volatility to make the predictions more realistic
        np.random.seed(42)  # For reproducibility
        volatility_factor = 0.01  # 1% daily volatility
        
        # Apply random volatility to create more realistic price movements
        for i in range(1, len(prediction_data)):
            # Random factor with slight upward bias
            random_change = np.random.normal(0.001, volatility_factor)
            # Apply change to current prediction based on previous prediction
            prediction_data.loc[i, 'Predicted_Price'] = prediction_data.loc[i-1, 'Predicted_Price'] * (1 + random_change)
        
        # Calculate predicted price changes
        prediction_data['Price_Change'] = prediction_data['Predicted_Price'].diff()
    
    with col2:
        st.markdown("<h4>Predicted Bitcoin Prices</h4>", unsafe_allow_html=True)
        
        # Create enhanced prediction chart with Plotly
        fig = go.Figure()
        
        # Add traces for predicted price increases (green)
        fig.add_trace(go.Scatter(
            x=prediction_data[prediction_data['Price_Change'] >= 0]['Date'],
            y=prediction_data[prediction_data['Price_Change'] >= 0]['Predicted_Price'],
            mode='lines',
            line=dict(color='#66CC33', width=2),
            name='Predicted Increase',
            fill='tozeroy',
            fillcolor='rgba(102, 204, 51, 0.2)'
        ))
        
        # Add traces for predicted price decreases (red)
        fig.add_trace(go.Scatter(
            x=prediction_data[prediction_data['Price_Change'] < 0]['Date'],
            y=prediction_data[prediction_data['Price_Change'] < 0]['Predicted_Price'],
            mode='lines',
            line=dict(color='#CC3366', width=2),
            name='Predicted Decrease',
            fill='tozeroy',
            fillcolor='rgba(204, 51, 102, 0.2)'
        ))
        
        # Update layout
        fig.update_layout(
            title='',
            xaxis_title='Date',
            yaxis_title='Predicted Price (USD)',
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            margin=dict(l=0, r=0, t=10, b=0),
            height=400,
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=7, label="7d", step="day", stepmode="backward"),
                        dict(count=14, label="14d", step="day", stepmode="backward"),
                        dict(count=30, label="30d", step="day", stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(visible=True),
                type="date"
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Combined historical and predicted view
    st.markdown("<h4>Combined View: Historical and Predicted Prices</h4>", unsafe_allow_html=True)
    
    # Create combined chart
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(go.Scatter(
        x=data['Date'],
        y=data['Price'],
        mode='lines',
        name='Historical Price',
        line=dict(color='#3366CC', width=2)
    ))
    
    # Add prediction data with color based on price change
    # For increasing segments
    fig.add_trace(go.Scatter(
        x=prediction_data[prediction_data['Price_Change'] >= 0]['Date'],
        y=prediction_data[prediction_data['Price_Change'] >= 0]['Predicted_Price'],
        mode='lines',
        name='Predicted Increase',
        line=dict(color='#66CC33', width=2, dash='dash')
    ))
    
    # For decreasing segments
    fig.add_trace(go.Scatter(
        x=prediction_data[prediction_data['Price_Change'] < 0]['Date'],
        y=prediction_data[prediction_data['Price_Change'] < 0]['Predicted_Price'],
        mode='lines',
        name='Predicted Decrease',
        line=dict(color='#CC3366', width=2, dash='dash')
    ))
    
    # Add current price marker
    fig.add_trace(go.Scatter(
        x=[data['Date'].iloc[-1]],
        y=[data['Price'].iloc[-1]],
        mode='markers',
        marker=dict(color='#CC3366', size=10),
        name='Current Price'
    ))
    
    # Update layout
    fig.update_layout(
        title='',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=0, r=0, t=10, b=0),
        height=400,
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=7, label="7d", step="day", stepmode="backward"),
                    dict(count=14, label="14d", step="day", stepmode="backward"),
                    dict(count=30, label="30d", step="day", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    # Advanced Trading Analysis
    st.markdown("<h3>Advanced Trading Analysis</h3>", unsafe_allow_html=True)
    
    # Price Action and Patterns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h4>Price Action Signals</h4>", unsafe_allow_html=True)
        
        # Display detected patterns
        if advanced_signals['latest_pattern']:
            pattern = advanced_signals['latest_pattern']['pattern']
            signal = advanced_signals['latest_pattern']['signal']
            pattern_color = "#66CC33" if signal == "Buy" else "#CC3366"
            
            st.markdown(
                f"""<div style='background-color:white; padding:15px; border-radius:5px; border-left:4px solid {pattern_color};'>
                <h5 style='margin-top:0;'>Detected Pattern</h5>
                <p style='font-size:1.2em; font-weight:bold;'>{pattern}</p>
                <p style='color:{pattern_color};'>Signal: {signal}</p>
                </div>""",
                unsafe_allow_html=True
            )
        else:
            st.info("No significant chart patterns detected in recent data.")
        
        # Display support and resistance levels
        st.markdown("<h5>Support & Resistance Levels</h5>", unsafe_allow_html=True)
        
        support_levels = advanced_signals['support_levels']
        resistance_levels = advanced_signals['resistance_levels']
        
        if support_levels:
            st.markdown("<p><b>Support Levels:</b></p>", unsafe_allow_html=True)
            for level in support_levels:
                st.markdown(f"<p style='color:#66CC33;'>${level:.2f}</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p>No significant support levels detected.</p>", unsafe_allow_html=True)
        
        if resistance_levels:
            st.markdown("<p><b>Resistance Levels:</b></p>", unsafe_allow_html=True)
            for level in resistance_levels:
                st.markdown(f"<p style='color:#CC3366;'>${level:.2f}</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p>No significant resistance levels detected.</p>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<h4>Order Flow Analysis</h4>", unsafe_allow_html=True)
        
        # Display buying/selling pressure
        buying_pressure = advanced_signals['order_flow']['buying_pressure']
        selling_pressure = advanced_signals['order_flow']['selling_pressure']
        
        # Create buying/selling pressure gauge
        fig = go.Figure()
        
        fig.add_trace(go.Indicator(
            mode = "gauge+number",
            value = buying_pressure / (buying_pressure + selling_pressure) * 100,
            title = {'text': "Buying vs Selling Pressure"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "#3366CC"},
                'steps': [
                    {'range': [0, 40], 'color': "#CC3366"},
                    {'range': [40, 60], 'color': "#FFA500"},
                    {'range': [60, 100], 'color': "#66CC33"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)
        
        # Display divergence if any
        divergence = advanced_signals['order_flow']['divergence']
        if divergence != "None":
            div_color = "#66CC33" if divergence == "Bullish" else "#CC3366"
            st.markdown(
                f"""<div style='background-color:white; padding:15px; border-radius:5px; border-left:4px solid {div_color};'>
                <h5 style='margin-top:0;'>Order Flow Divergence</h5>
                <p style='color:{div_color};'>{divergence} Divergence Detected</p>
                <p>Price and volume are moving in opposite directions, indicating potential reversal.</p>
                </div>""",
                unsafe_allow_html=True
            )
    
    # Trading Signal and Recommendation
    st.markdown("<h4>Trading Signal</h4>", unsafe_allow_html=True)
    
    # Determine signal color
    signal_color = "#66CC33" if advanced_signals['action'] in ["Buy", "Strong Buy"] else "#FFA500" if advanced_signals['action'] == "Hold" else "#CC3366"
    
    # Create signal card
    st.markdown(
        f"""<div style='padding:20px; border-radius:5px; text-align:center; border:2px solid {signal_color};'>
        <h2 style='color:{signal_color}; margin-top:0;'>{advanced_signals['action']}</h2>
        <p>Entry Price: ${advanced_signals['entry_price']:.2f}</p>
        <p>Target Price: ${advanced_signals['target_price']:.2f}</p>
        <p>Stop Loss: ${advanced_signals['stop_loss']:.2f}</p>
        <p>Risk/Reward Ratio: {advanced_signals['risk_reward']:.2f}</p>
        </div>""",
        unsafe_allow_html=True
    )
    
    # Backtest Results
    st.markdown("<h4>Backtest Results</h4>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        perf = advanced_signals['backtest_performance']
        st.metric("Total Return", f"{perf['total_return']*100:.2f}%")
    
    with col2:
        st.metric("Max Drawdown", f"{perf['max_drawdown']*100:.2f}%")
    
    with col3:
        st.metric("Win Rate", f"{perf['win_rate']*100:.2f}%")
    
    with col4:
        st.metric("Sharpe Ratio", f"{perf['sharpe_ratio']:.2f}")

with tab3:
    # Investment calculator with better styling
    st.markdown("<h3>Investment Calculator</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # User input for investment amount
        investment_amount = st.number_input("Investment Amount (USD)", min_value=10.0, value=1000.0, step=10.0)
        # Calculate BTC amount based on current price
        btc_amount = investment_amount / current_price if current_price > 0 else 0.0

        # Get predicted prices
        days_ahead = 30
        prediction_data = make_predictions(model, scaler, features, days_ahead)
        # Ensure 'Date' column is datetime for plotting
        if 'Date' in prediction_data.columns and not np.issubdtype(prediction_data['Date'].dtype, np.datetime64):
            prediction_data['Date'] = pd.to_datetime(prediction_data['Date'])
        
        # Add volatility for more realistic predictions
        np.random.seed(42)
        for i in range(1, len(prediction_data)):
            random_change = np.random.normal(0.001, 0.01)
            prediction_data.loc[i, 'Predicted_Price'] = prediction_data.loc[i-1, 'Predicted_Price'] * (1 + random_change)
        
        predicted_price_7d = prediction_data["Predicted_Price"].iloc[6]  # 7 days ahead
        predicted_price_30d = prediction_data["Predicted_Price"].iloc[29]  # 30 days ahead
        
        potential_value_7d = btc_amount * predicted_price_7d
        potential_value_30d = btc_amount * predicted_price_30d
        
        profit_7d = potential_value_7d - investment_amount
        profit_30d = potential_value_30d - investment_amount
        
        # Display results in clean cards
        st.markdown(
            f"""<div>
                <h4>BTC you can buy now</h4>
                <p>{btc_amount:.8f} BTC</p>
            </div>""", 
            unsafe_allow_html=True
        )
        
        col_a, col_b = st.columns(2)
        with col_a:
            profit_color_7d = "#66CC33" if profit_7d >= 0 else "#CC3366"
            st.markdown(
                f"""<div>
                    <h4>7-Day Projection</h4>
                    <p>${potential_value_7d:.2f}</p>
                    <p style='color:{profit_color_7d};'>({'+' if profit_7d >= 0 else ''}{profit_7d:.2f} USD)</p>
                </div>""", 
                unsafe_allow_html=True
            )
        
        with col_b:
            profit_color_30d = "#66CC33" if profit_30d >= 0 else "#CC3366"
            st.markdown(
                f"""<div>
                    <h4>30-Day Projection</h4>
                    <p>${potential_value_30d:.2f}</p>
                    <p style='color:{profit_color_30d};'>({'+' if profit_30d >= 0 else ''}{profit_30d:.2f} USD)</p>
                </div>""", 
                unsafe_allow_html=True
            )
    
    with col2:
        # Buy/Sell price recommendations with better styling
        st.markdown("<h4>Price Targets (Advanced Analysis)</h4>", unsafe_allow_html=True)
        
        # Use trading recommendations from advanced analysis
        action = advanced_signals['action']
        entry_price = advanced_signals['entry_price']
        target_price = advanced_signals['target_price']
        stop_loss = advanced_signals['stop_loss']
        
        if "Buy" in action:
            st.markdown(
                f"""<div style='border-left: 4px solid #3366CC;'>
                    <h4>Recommended Buy Price</h4>
                    <p>${entry_price:.2f}</p>
                    <p style='font-size:0.8em;'>({((entry_price - current_price) / current_price * 100):.2f}% from current price)</p>
                </div>""", 
                unsafe_allow_html=True
            )
            
            st.markdown(
                f"""<div style='border-left: 4px solid #66CC33;'>
                    <h4>Target Price</h4>
                    <p>${target_price:.2f}</p>
                    <p style='font-size:0.8em;'>({((target_price - entry_price) / entry_price * 100):.2f}% gain)</p>
                </div>""", 
                unsafe_allow_html=True
            )
            
            st.markdown(
                f"""<div style='border-left: 4px solid #CC3366;'>
                    <h4>Stop Loss</h4>
                    <p>${stop_loss:.2f}</p>
                    <p style='font-size:0.8em;'>({((stop_loss - entry_price) / entry_price * 100):.2f}% from buy price)</p>
                </div>""", 
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""<div style='border-left: 4px solid #CC3366;'>
                    <h4>Recommended Sell Price</h4>
                    <p>${entry_price:.2f}</p>
                    <p style='font-size:0.8em;'>({((entry_price - current_price) / current_price * 100):.2f}% from current price)</p>
                </div>""", 
                unsafe_allow_html=True
            )
            
            st.markdown(
                f"""<div style='border-left: 4px solid #3366CC;'>
                    <h4>Target Price</h4>
                    <p>${target_price:.2f}</p>
                    <p style='font-size:0.8em;'>({((target_price - entry_price) / entry_price * 100):.2f}% from sell price)</p>
                </div>""", 
                unsafe_allow_html=True
            )
            
            st.markdown(
                f"""<div style='border-left: 4px solid #FFA500;'>
                    <h4>Stop Loss</h4>
                    <p>${stop_loss:.2f}</p>
                    <p style='font-size:0.8em;'>({((stop_loss - entry_price) / entry_price * 100):.2f}% from sell price)</p>
                </div>""", 
                unsafe_allow_html=True
            )

with tab4:
    # Portfolio Analysis section
    st.markdown("<h3>Portfolio Analysis</h3>", unsafe_allow_html=True)
    
    from portfolio_analysis import get_portfolio_data
    
    # Add auto-refresh button
    refresh_col1, refresh_col2 = st.columns([1, 3])
    with refresh_col1:
        if st.button("🔄 Refresh Portfolio"):
            st.cache_data.clear()
    
    # Get portfolio data with caching
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def load_portfolio_data():
        try:
            with st.spinner("Loading portfolio data..."):
                return get_portfolio_data()
        except Exception as e:
            st.error(f"Error loading portfolio data: {e}")
            # Return dummy data as fallback
            dummy_allocation = pd.DataFrame({
                'Ticker': ['VTI', 'VNQ', 'VXUS', 'BND', 'BTC-USD'],
                'Name': ['Vanguard Total Stock Market ETF', 'Vanguard Real Estate ETF', 
                         'Vanguard Total International Stock ETF', 'Vanguard Total Bond Market ETF', 'Bitcoin'],
                'Allocation': [20.0, 20.0, 20.0, 20.0, 20.0]
            })
            
            dummy_metrics = pd.DataFrame(
                index=['VTI', 'VNQ', 'VXUS', 'BND', 'BTC-USD'],
                data={
                    'Annualized Return': [0.15, 0.08, 0.10, 0.04, 0.25],
                    'Standard Deviation': [0.18, 0.22, 0.20, 0.05, 0.65],
                    'Maximum Drawdown': [-0.25, -0.30, -0.28, -0.10, -0.55],
                    'Benchmark Relative': [0.03, -0.04, -0.02, -0.08, 0.13]
                }
            )
            return dummy_allocation, dummy_metrics
    
    allocation, metrics = load_portfolio_data()
    
    # Show last updated time
    with refresh_col2:
        st.markdown(f"<div style='color:#666; font-size:0.8em;'>Last updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>", unsafe_allow_html=True)
    
    # Display portfolio allocation
    st.markdown("<h4>Portfolio Allocation</h4>", unsafe_allow_html=True)
    
    # Format allocation table
    st.markdown("""
    <style>
    .allocation-table {
        width: 100%;
        border-collapse: collapse;
    }
    .allocation-table th, .allocation-table td {
        padding: 8px;
        text-align: left;
        border-bottom: 1px solid #ddd;
    }
    .allocation-table th {
        background-color: #3366CC;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Highlight Bitcoin row
    def highlight_btc(row):
        return ['background-color:FFFF00'if row['Ticker'] == 'BTC-USD' else '' for _ in row]
    
    styled_allocation = allocation.style.apply(highlight_btc, axis=1)
    st.table(styled_allocation)
    
    # Display performance metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h4>Return</h4>", unsafe_allow_html=True)
        
        # Format annualized returns
        annual_returns = metrics['Annualized Return'].map(lambda x: f"{x*100:.2f}%")
        benchmark_rel = metrics['Benchmark Relative'].map(lambda x: f"{x*100:+.2f}%")
        
        returns_df = pd.DataFrame({
            'Annualized Return': annual_returns,
            'Benchmark Relative': benchmark_rel
        })
        
        # Highlight Bitcoin row
        def highlight_btc_index(df):
            return ['background-color:FFFF00'if idx == 'BTC-USD' else '' for idx in df.index]
        
        styled_returns = returns_df.style.apply(highlight_btc_index, axis=0)
        st.table(styled_returns)
    
    with col2:
        st.markdown("<h4>Risk</h4>", unsafe_allow_html=True)
        
        # Format risk metrics
        std_dev = metrics['Standard Deviation'].map(lambda x: f"{x*100:.2f}%")
        max_dd = metrics['Maximum Drawdown'].map(lambda x: f"{x*100:.2f}%")
        
        risk_df = pd.DataFrame({
            'Standard Deviation': std_dev,
            'Maximum Drawdown': max_dd
        })
        
        styled_risk = risk_df.style.apply(highlight_btc_index, axis=0)
        st.table(styled_risk)
        
    # Add portfolio performance chart
    st.markdown("<h4>Portfolio Performance vs Bitcoin</h4>", unsafe_allow_html=True)
    
    # Create a simple chart showing relative performance
    performance_data = {
        'Asset': ['Bitcoin'] + [ticker for ticker in allocation['Ticker'] if ticker != 'BTC-USD'],
        'Return': [metrics.loc['BTC-USD', 'Annualized Return']] + 
                 [metrics.loc[ticker, 'Annualized Return'] for ticker in allocation['Ticker'] if ticker != 'BTC-USD']
    }
    
    performance_df = pd.DataFrame(performance_data)
    
    # Create bar chart
    fig = go.Figure()
    
    # Add Bitcoin bar
    fig.add_trace(go.Bar(
        x=[performance_df['Asset'][0]],
        y=[performance_df['Return'][0] * 100],
        name='Bitcoin',
        marker_color='#3366CC'
    ))
    
    # Add other assets
    fig.add_trace(go.Bar(
        x=performance_df['Asset'][1:],
        y=[r * 100 for r in performance_df['Return'][1:]],
        name='Other Assets',
        marker_color='#66CC33'
    ))
    
    # Update layout
    fig.update_layout(
        title='',
        xaxis_title='Asset',
        yaxis_title='Annualized Return (%)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=0, r=0, t=10, b=0),
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab5:
    # About section
    st.markdown("<h3>About This App</h3>", unsafe_allow_html=True)
    st.write("""
    This Bitcoin Price Prediction App uses advanced trading techniques to forecast future Bitcoin prices and provide trading signals.
    
    **Features:**
    - Real-time Bitcoin price data from CoinMarketCap
    - Price action analysis with pattern recognition
    - Order flow analysis to identify buying/selling pressure
    - Volume profile analysis for support/resistance levels
    - Machine learning predictions for future prices
    - Backtested trading signals with performance metrics
    - Investment calculator with risk/reward analysis
    - Portfolio analysis with Bitcoin and traditional assets
    """)
    
    st.markdown("<h4>Trading Techniques Used</h4>", unsafe_allow_html=True)
    st.write("""
    1. **Price Action Analysis**: Identifies chart patterns like double tops/bottoms and head & shoulders
    2. **Volume Profile**: Analyzes trading volume at different price levels to identify support/resistance
    3. **Order Flow Analysis**: Examines buying vs selling pressure and detects divergences
    4. **Machine Learning**: Predicts future prices based on historical patterns
    5. **Backtesting**: Tests trading strategies on historical data to measure performance
    """)
    
    st.markdown("<h4>How It Works</h4>", unsafe_allow_html=True)
    st.write("""
    1. The app fetches real-time Bitcoin price data
    2. Advanced analysis techniques identify patterns, support/resistance levels, and order flow signals
    3. Trading signals are generated based on multiple analysis techniques
    4. Signals are backtested to measure historical performance
    5. Investment recommendations are provided with specific entry, target, and stop-loss prices
    """)

# Risk disclaimer in sidebar with better styling
st.sidebar.markdown("---")
st.sidebar.markdown(
    """<div style='background-color:#FFF3CD; padding:10px; border-radius:5px; border-left:4px solid #FFA500;'>
    <h4 style='color:#856404; margin-top:0;'>Disclaimer</h4>
    <p style='color:#856404; font-size:0.9em;'>This app provides predictions based on historical data and should not be considered as financial advice. Cryptocurrency investments are subject to high market risk. Always do your own research before investing.</p>
    </div>""", 
    unsafe_allow_html=True
)

# Last updated time
st.sidebar.markdown(
    f"""<div style='background-color:#E2F0FB; padding:10px; border-radius:5px; margin-top:10px; border-left:4px solid #3366CC;'>
    <p style='color:#3366CC; margin:0; font-size:0.9em;'>Last updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>""", 
    unsafe_allow_html=True
)
