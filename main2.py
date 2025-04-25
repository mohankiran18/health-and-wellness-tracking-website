import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import os

def create_dataset(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:(i + window_size)])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def run_forecast(filepath):
    df = pd.read_csv(filepath)
    prices = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_prices = scaler.fit_transform(prices)

    window_sizes = [3, 6, 10]
    predictions = []
    models = []

    for window in window_sizes:
        X, y = create_dataset(scaled_prices, window)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        model = build_lstm_model((X.shape[1], 1))
        model.fit(X, y, epochs=10, batch_size=16, verbose=0)
        pred = model.predict(X)
        pred = scaler.inverse_transform(pred)
        predictions.append(pred)
        models.append(model)

    # Ensure all predictions have the same length (use shortest)
    min_len = min(pred.shape[0] for pred in predictions)
    predictions = [pred[:min_len] for pred in predictions]

    # Average the predictions
    avg_prediction = np.mean(predictions, axis=0)

    # Truncate actual values to match prediction length
    max_window = max(window_sizes)
    actual_start = max_window + (len(prices[max_window:]) - min_len)
    actual = prices[actual_start:]

    mse = mean_squared_error(actual, avg_prediction)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, avg_prediction)
    r2 = r2_score(actual, avg_prediction)

    # Save plot
    plt.figure(figsize=(10, 5))
    plt.plot(actual, label='Actual Prices')
    plt.plot(avg_prediction, label='Predicted Prices')
    plt.title('Bagged LSTM Forecasting')
    plt.legend()
    plot_path = 'static/prediction_plot.png'
    plt.savefig(os.path.join('static', 'prediction_plot.png'))
    plt.close()

    return {'MSE': round(mse, 2), 'RMSE': round(rmse, 2), 'MAE': round(mae, 2), 'R2': round(r2, 4)}, plot_path
