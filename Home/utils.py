import os
import time
import numpy as np
import pandas as pd
from alpha_vantage.timeseries import TimeSeries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Step 1: Data Collection
def fetch_stock_data(symbol):
    api_key = 'H2426D083MH6N3F5'
    ts = TimeSeries(key=api_key, output_format='pandas')
    data, _ = ts.get_daily(symbol=symbol, outputsize='full')
    return data

# Step 2: Data Preprocessing and Feature Engineering
def preprocess_data(data):
    target = data['4. close']
    features = data.drop(columns=['4. close', '5. volume'])
    return features, target

# Step 3: Model Selection and Training
def train_model(features, target, epochs=50):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(features)

    X = np.reshape(scaled_features, (scaled_features.shape[0], 1, scaled_features.shape[1]))
    X_train, X_test, y_train, y_test = train_test_split(X, target.values, test_size=0.2, random_state=42)

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(1, X_train.shape[2])))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_split=0.1, verbose=1)

    return model, X_test, y_test, scaler

# Step 4: Real-time Prediction
def predict_stock_price(model, features, scaler):
    predicted_prices = model.predict(features)
    predicted_prices = scaler.inverse_transform(predicted_prices)
    return predicted_prices

# Step 5: Visualization
def visualize_results_and_save(dates, actual_prices, predicted_prices):
    plt.figure(figsize=(15, 7))
    plt.plot(dates, actual_prices, label='Actual Prices', color='blue')
    plt.plot(dates, predicted_prices, label='Predicted Prices', color='red', linestyle='dashed')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Actual vs. Predicted Stock Prices')
    plt.legend()

    image_path = 'predicted_graph.png'
    plt.savefig(os.path.join('D:/stock_price_predictor/myspace/predictor/static/', image_path))
    plt.close()

    return image_path

def main(symbol, prediction_days):
    stock_data = fetch_stock_data(symbol)
    features, target = preprocess_data(stock_data)
    model, X_test, y_test, scaler = train_model(features, target)

    for _ in range(prediction_days):
        new_data = fetch_stock_data(symbol)
        new_features, new_target = preprocess_data(new_data)
        scaled_new_features = scaler.transform(new_features)
        X_new = np.reshape(scaled_new_features, (scaled_new_features.shape[0], 1, scaled_new_features.shape[1]))

        predicted_price = predict_stock_price(model, X_new, scaler)
        print("Predicted Price for", symbol, ":", predicted_price[-1][0])

        visualize_results_and_save(new_target.index, new_target.values, predicted_price)

        time.sleep(300)

# Entry point of the script
if __name__ == "__main__":
    symbol = input("Enter the stock symbol (e.g., AAPL): ")
    prediction_days = int(input("Enter the number of days for prediction: "))
    main(symbol, prediction_days)
