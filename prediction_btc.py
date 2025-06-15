import pandas as pd
import numpy as np
import datetime

def make_predictions(model, scaler, features, days_ahead=30, last_prices=None):
    # Create future dates
    future_dates = pd.date_range(start=datetime.datetime.now(), periods=days_ahead)
    
    # Create dataframe for predictions
    future_df = pd.DataFrame({"Date": future_dates})
    future_df['DayOfWeek'] = future_df['Date'].dt.dayofweek
    future_df['Month'] = future_df['Date'].dt.month
    future_df['Day'] = future_df['Date'].dt.day
    
    # Initialize with the last known prices if provided
    if last_prices is not None and len(last_prices) >= 5:
        for i in range(1, 6):
            future_df[f'Price_Lag_{i}'] = last_prices[-i]
    else:
        # Use placeholder values
        for i in range(1, 6):
            future_df[f'Price_Lag_{i}'] = 0
    
    # Make predictions one day at a time
    predictions = []
    
    for i in range(days_ahead):
        # For the first prediction, we use historical data
        # For subsequent predictions, we use previous predictions
        if i >= 1:
            # Update lag features based on previous predictions
            for j in range(min(i, 5), 0, -1):
                future_df.loc[i, f'Price_Lag_{j}'] = scaled_pred if j == 1 else future_df.loc[i-j+1, f'Price_Lag_{j-1}']
        
        # Get features for prediction
        X_pred = future_df.loc[i:i, features].values
        
        # Make prediction
        scaled_pred = model.predict(X_pred)[0]
        predictions.append(scaled_pred)
    
    # Convert scaled predictions back to original scale
    predictions_array = np.array(predictions).reshape(-1, 1)
    price_predictions = scaler.inverse_transform(predictions_array).flatten()
    
    # Create prediction dataframe
    prediction_data = pd.DataFrame({
        "Date": future_dates,
        "Predicted_Price": price_predictions
    })
    
    return prediction_data
