from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

def preprocess_data(data):
    # Add features: day of week, month, etc.
    data = data.copy()
    data['DayOfWeek'] = data['Date'].dt.dayofweek
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day
    
    # Create lag features (previous days' prices)
    for i in range(1, 6):  # 5 days of lag features
        data[f'Price_Lag_{i}'] = data['Price'].shift(i)
    
    # Drop rows with NaN values from lag features
    data = data.dropna()
    
    # Scale price data
    scaler = MinMaxScaler()
    data["Scaled_Price"] = scaler.fit_transform(data["Price"].values.reshape(-1, 1))
    
    return data, scaler

def train_model(data):
    # Features: use date components and lag features
    features = ['DayOfWeek', 'Month', 'Day'] + [f'Price_Lag_{i}' for i in range(1, 6)]
    X = data[features].values
    y = data["Scaled_Price"].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    return model, mse, features