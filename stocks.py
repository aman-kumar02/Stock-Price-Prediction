import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Function to get stock data
def get_stock_data(ticker):
    stock_data = yf.download(ticker, period="1mo", interval="5m")
    if stock_data.empty:
        print("No data available for the given ticker.")
        return pd.DataFrame()
    return stock_data


# Function to preprocess data
def preprocess_data(df):
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['Volatility'] = df['Close'].rolling(window=10).std()
    df = df.dropna()

    X = df[['Open', 'High', 'Low', 'Volume', 'MA10', 'MA50', 'Volatility']]
    y = df['Close']
    return X, y


# Function to train the model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
    return model, scaler


# Function to predict
def predict_next_close(model, scaler, latest_data):
    latest_data = scaler.transform([latest_data])
    prediction = model.predict(latest_data)
    return prediction[0]


# Function to plot stock prices
def plot_stock_prices(df, prediction, next_interval_time):
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['Close'], label='Actual Close Prices', color='blue')
    plt.axvline(x=next_interval_time, color='red', linestyle='--', label='Prediction Time')
    plt.scatter(next_interval_time, prediction, color='red', label=f'Predicted Close Price: {prediction:.2f}')
    plt.title('Stock Prices with Predicted Next Close Price')
    plt.xlabel('Date and Time')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()


# Main function
if __name__ == "__main__":
    ticker = "NHPC.NS"  # Example ticker for Reliance Industries on NSE

    # Get data
    stock_data = get_stock_data(ticker)
    if stock_data.empty:
        print("No data available for the given ticker.")
    else:
        print("Data fetched successfully.")

        # Preprocess data
        X, y = preprocess_data(stock_data)
        if X.empty or y.empty:
            print("No data available after preprocessing.")
        else:
            print("Data preprocessed successfully.")

            # Train model
            model, scaler = train_model(X, y)
            print("Model trained successfully.")

            # Predict next close
            latest_data = X.iloc[-1].values  # Using the last row of the feature set for prediction
            prediction = predict_next_close(model, scaler, latest_data)
            next_interval_time = X.index[-1] + pd.Timedelta(minutes=5)
            print(f"Predicted next close price at {next_interval_time}: {prediction}")

            # Plot stock prices with prediction
            plot_stock_prices(stock_data, prediction, next_interval_time)

# import pandas as pd
# import yfinance as yf
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
#
#
# # Function to get stock data
# def get_stock_data(ticker):
#     stock_data = yf.download(ticker, period="1mo", interval="5m")
#     if stock_data.empty:
#         print("No data available for the given ticker.")
#         return pd.DataFrame()
#     return stock_data
#
#
# # Function to calculate MACD
# def calculate_macd(df):
#     df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
#     df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
#     df['MACD'] = df['EMA12'] - df['EMA26']
#     df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
#     return df
#
#
# # Function to preprocess data
# def preprocess_data(df):
#     df['MA10'] = df['Close'].rolling(window=10).mean()
#     df['MA50'] = df['Close'].rolling(window=50).mean()
#     df['Volatility'] = df['Close'].rolling(window=10).std()
#     df = calculate_macd(df)
#     df = df.dropna()
#
#     X = df[['Open', 'High', 'Low', 'Volume', 'MA10', 'MA50', 'Volatility', 'MACD', 'Signal_Line']]
#     y = df['Close']
#     return X, y
#
#
# # Function to train the model
# def train_model(X, y):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)
#
#     model = RandomForestRegressor(n_estimators=100, random_state=42)
#     model.fit(X_train, y_train)
#
#     y_pred = model.predict(X_test)
#     mse = mean_squared_error(y_test, y_pred)
#     print(f'Mean Squared Error: {mse}')
#     return model, scaler
#
#
# # Function to predict
# def predict_next_close(model, scaler, latest_data):
#     latest_data = scaler.transform([latest_data])
#     prediction = model.predict(latest_data)
#     return prediction[0]
#
#
# # Function to plot stock prices with MACD
# def plot_stock_prices(df, prediction, next_interval_time):
#     fig, (ax1, ax2) = plt.subplots(2, figsize=(14, 10), sharex=True)
#
#     # Plotting the actual close prices and prediction
#     ax1.plot(df.index, df['Close'], label='Actual Close Prices', color='blue')
#     ax1.axvline(x=next_interval_time, color='red', linestyle='--', label='Prediction Time')
#     ax1.scatter(next_interval_time, prediction, color='red', label=f'Predicted Close Price: {prediction:.2f}')
#     ax1.set_title('Stock Prices with Predicted Next Close Price')
#     ax1.set_ylabel('Close Price')
#     ax1.legend()
#
#     # Plotting the MACD and Signal Line
#     ax2.plot(df.index, df['MACD'], label='MACD', color='green')
#     ax2.plot(df.index, df['Signal_Line'], label='Signal Line', color='orange')
#     ax2.set_title('MACD Indicator')
#     ax2.set_xlabel('Date and Time')
#     ax2.set_ylabel('MACD')
#     ax2.legend()
#
#     plt.show()
#
#
# # Main function
# if __name__ == "__main__":
#     ticker = "NHPC.NS"  # Example ticker for Reliance Industries on NSE
#
#     # Get data
#     stock_data = get_stock_data(ticker)
#     if stock_data.empty:
#         print("No data available for the given ticker.")
#     else:
#         print("Data fetched successfully.")
#
#         # Preprocess data
#         X, y = preprocess_data(stock_data)
#         if X.empty or y.empty:
#             print("No data available after preprocessing.")
#         else:
#             print("Data preprocessed successfully.")
#
#             # Train model
#             model, scaler = train_model(X, y)
#             print("Model trained successfully.")
#
#             # Predict next close
#             latest_data = X.iloc[-1].values  # Using the last row of the feature set for prediction
#             prediction = predict_next_close(model, scaler, latest_data)
#             next_interval_time = X.index[-1] + pd.Timedelta(minutes=5)
#             print(f"Predicted next close price at {next_interval_time}: {prediction}")
#
#             # Plot stock prices with prediction and MACD
#             plot_stock_prices(stock_data, prediction, next_interval_time)
