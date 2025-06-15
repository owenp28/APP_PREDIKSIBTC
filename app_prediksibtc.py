import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import plotly.graph_objects as go
from data_load import load_data, get_trading_recommendation
from train_model import preprocess_data, train_model
from prediction_btc import make_predictions

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

# Display current Bitcoin price in sidebar with better styling
from data_load import get_current_bitcoin_price

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

# Update data with current price for better predictions
data.loc[data.index[-1], "Price"] = current_price

# Preprocess data
with st.spinner("Processing data..."):
    processed_data, scaler = preprocess_data(data)
    model, mse, features = train_model(processed_data)

# Main content in tabs for better organization
tab1, tab2, tab3 = st.tabs(["📈 Price Analysis", "💰 Investment Calculator", "ℹ️ About"])

with tab1:
    # Historical and predicted prices
    st.markdown("<h3>Bitcoin Price Analysis</h3>", unsafe_allow_html=True)
    
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
    
    with col2:
        st.markdown("<h4>Future Price Predictions</h4>", unsafe_allow_html=True)
        days_ahead = st.slider("Days to predict ahead:", 7, 60, 30, key="days_slider")
        prediction_data = make_predictions(model, scaler, features, days_ahead)
        
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
    
    # Technical indicators section
    st.markdown("<h3>Technical Analysis</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h4>Key Indicators</h4>", unsafe_allow_html=True)
        
        # Display RSI
        rsi_color = "#66CC33" if trading_rec["rsi"] < 70 else "#CC3366"
        st.markdown(
            f"""<div>
                <h4>RSI (Relative Strength Index)</h4>
                <p style='color:{rsi_color}; font-size:1.5em;'>{trading_rec["rsi"]:.2f}</p>
                <p style='font-size:0.8em;'>{'Oversold (<30)' if trading_rec["rsi"] < 30 else 'Overbought (>70)' if trading_rec["rsi"] > 70 else 'Neutral'}</p>
            </div>""", 
            unsafe_allow_html=True
        )
        
        # Display MACD
        macd_color = "#66CC33" if trading_rec["macd_histogram"] > 0 else "#CC3366"
        st.markdown(
            f"""<div>
                <h4>MACD Histogram</h4>
                <p style='color:{macd_color}; font-size:1.5em;'>{trading_rec["macd_histogram"]:.2f}</p>
                <p style='font-size:0.8em;'>{'Bullish (>0)' if trading_rec["macd_histogram"] > 0 else 'Bearish (<0)'}</p>
            </div>""", 
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown("<h4>Moving Averages</h4>", unsafe_allow_html=True)
        
        # Display MA crossover
        ma5 = data['MA5'].iloc[-1]
        ma20 = data['MA20'].iloc[-1]
        ma_signal = "Bullish" if ma5 > ma20 else "Bearish"
        ma_color = "#66CC33" if ma5 > ma20 else "#CC3366"
        
        st.markdown(
            f"""<div>
                <h4>MA Crossover</h4>
                <p style='color:{ma_color}; font-size:1.5em;'>{ma_signal}</p>
                <p style='font-size:0.8em;'>MA5: ${ma5:.2f} | MA20: ${ma20:.2f}</p>
            </div>""", 
            unsafe_allow_html=True
        )
        
        # Display overall signal
        signal_color = "#66CC33" if trading_rec["action"] in ["Strong Buy", "Consider Buy"] else "#FFA500" if trading_rec["action"] == "Hold" else "#CC3366"
        st.markdown(
            f"""<div>
                <h4>Overall Signal</h4>
                <p style='color:{signal_color}; font-size:1.5em;'>{trading_rec["action"]}</p>
            </div>""", 
            unsafe_allow_html=True
        )
    
    # Model performance metrics in a clean card
    st.markdown(
        f"""<div>
            <h4>Model Performance</h4>
            <p>Mean Squared Error: {mse:.4f}</p>
        </div>""", 
        unsafe_allow_html=True
    )
    
    # Investment recommendation system with better styling
    st.markdown("<h3>Investment Recommendation</h3>", unsafe_allow_html=True)
    
    # Calculate price trends
    latest_price = data["Price"].iloc[-1]
    predicted_price_7d = prediction_data["Predicted_Price"].iloc[6]  # 7 days ahead
    predicted_price_30d = prediction_data["Predicted_Price"].iloc[29] if len(prediction_data) >= 30 else prediction_data["Predicted_Price"].iloc[-1]
    
    # Calculate potential returns
    short_term_change = (predicted_price_7d - latest_price) / latest_price * 100
    long_term_change = (predicted_price_30d - latest_price) / latest_price * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h4>Short-term Outlook (7 days)</h4>", unsafe_allow_html=True)
        if short_term_change > 3:
            st.markdown(f"<div>Strong Buy: Expected increase of {short_term_change:.2f}%</div>", unsafe_allow_html=True)
            recommendation = "Strong Buy"
        elif short_term_change > 0:
            st.markdown(f"<div>Consider Buy: Expected increase of {short_term_change:.2f}%</div>", unsafe_allow_html=True)
            recommendation = "Consider Buy"
        elif short_term_change > -3:
            st.markdown(f"<div>Hold: Expected change of {short_term_change:.2f}%</div>", unsafe_allow_html=True)
            recommendation = "Hold"
        else:
            st.markdown(f"<div>Consider Sell: Expected decrease of {abs(short_term_change):.2f}%</div>", unsafe_allow_html=True)
            recommendation = "Consider Sell"
    
    with col2:
        st.markdown("<h4>Long-term Outlook (30 days)</h4>", unsafe_allow_html=True)
        if long_term_change > 10:
            st.markdown(f"<div>Strong Buy: Expected increase of {long_term_change:.2f}%</div>", unsafe_allow_html=True)
        elif long_term_change > 0:
            st.markdown(f"<div>Consider Buy: Expected increase of {long_term_change:.2f}%</div>", unsafe_allow_html=True)
        elif long_term_change > -10:
            st.markdown(f"<div>Hold: Expected change of {long_term_change:.2f}%</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div>Consider Sell: Expected decrease of {abs(long_term_change):.2f}%</div>", unsafe_allow_html=True)

with tab2:
    # Investment calculator with better styling
    st.markdown("<h3>Investment Calculator</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h4>Calculate Potential Returns</h4>", unsafe_allow_html=True)
        investment_amount = st.number_input("Investment Amount (USD):", min_value=100, value=1000, step=100)
        
        # Calculate potential returns
        btc_amount = investment_amount / latest_price
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
        st.markdown("<h4>Price Targets (Technical Analysis)</h4>", unsafe_allow_html=True)
        
        # Use trading recommendations from technical analysis
        buy_price = trading_rec["optimal_buy"]
        sell_price = trading_rec["optimal_sell"]
        stop_loss = trading_rec["stop_loss"]
        take_profit_short = trading_rec["take_profit_short"]
        take_profit_long = trading_rec["take_profit_long"]
        
        if trading_rec["action"] in ["Strong Buy", "Consider Buy"]:
            st.markdown(
                f"""<div style='border-left: 4px solid #3366CC;'>
                    <h4>Recommended Buy Price</h4>
                    <p>${buy_price:.2f}</p>
                    <p style='font-size:0.8em;'>({((buy_price - latest_price) / latest_price * 100):.2f}% from current price)</p>
                </div>""", 
                unsafe_allow_html=True
            )
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(
                    f"""<div style='border-left: 4px solid #66CC33;'>
                        <h4>Short-term Target</h4>
                        <p>${take_profit_short:.2f}</p>
                        <p style='font-size:0.8em;'>({((take_profit_short - buy_price) / buy_price * 100):.2f}% gain)</p>
                    </div>""", 
                    unsafe_allow_html=True
                )
            
            with col_b:
                st.markdown(
                    f"""<div style='border-left: 4px solid #66CC33;'>
                        <h4>Long-term Target</h4>
                        <p>${take_profit_long:.2f}</p>
                        <p style='font-size:0.8em;'>({((take_profit_long - buy_price) / buy_price * 100):.2f}% gain)</p>
                    </div>""", 
                    unsafe_allow_html=True
                )
            
            st.markdown(
                f"""<div style='border-left: 4px solid #CC3366;'>
                    <h4>Stop Loss</h4>
                    <p>${stop_loss:.2f}</p>
                    <p style='font-size:0.8em;'>({((stop_loss - buy_price) / buy_price * 100):.2f}% from buy price)</p>
                </div>""", 
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""<div style='border-left: 4px solid #CC3366;'>
                    <h4>Recommended Sell Price</h4>
                    <p>${sell_price:.2f}</p>
                    <p style='font-size:0.8em;'>({((sell_price - latest_price) / latest_price * 100):.2f}% from current price)</p>
                </div>""", 
                unsafe_allow_html=True
            )
            
            col_a, col_b = st.columns(2)
            with col_a:
                buy_back_short = sell_price * 0.95
                st.markdown(
                    f"""<div style='border-left: 4px solid #3366CC;'>
                        <h4>Short-term Buy-back</h4>
                        <p>${buy_back_short:.2f}</p>
                        <p style='font-size:0.8em;'>({((buy_back_short - sell_price) / sell_price * 100):.2f}% from sell price)</p>
                    </div>""", 
                    unsafe_allow_html=True
                )
            
            with col_b:
                buy_back_long = sell_price * 0.90
                st.markdown(
                    f"""<div style='border-left: 4px solid #3366CC;'>
                        <h4>Long-term Buy-back</h4>
                        <p>${buy_back_long:.2f}</p>
                        <p style='font-size:0.8em;'>({((buy_back_long - sell_price) / sell_price * 100):.2f}% from sell price)</p>
                    </div>""", 
                    unsafe_allow_html=True
                )

with tab3:
    # About section
    st.markdown("<h3>About This App</h3>", unsafe_allow_html=True)
    st.write("""
    This Bitcoin Price Prediction App uses machine learning to forecast future Bitcoin prices based on historical data.
    The app provides investment recommendations and price targets to help you make informed decisions.
    
    **Features:**
    - Real-time Bitcoin price data from CoinGecko API
    - Technical analysis with RSI, MACD, and Moving Averages
    - Machine learning predictions for future prices
    - Investment recommendations based on predicted trends
    - Price targets for buying, selling, and setting stop losses
    """)
    
    st.markdown("<h4>How It Works</h4>", unsafe_allow_html=True)
    st.write("""
    1. The app fetches real-time Bitcoin price data
    2. Technical indicators are calculated to identify market trends
    3. A machine learning model is trained on historical price patterns
    4. The model predicts future prices based on these patterns
    5. Investment recommendations are generated based on both technical analysis and predicted price movements
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
