# Stock Price Prediction using Machine Learning

This project utilizes machine learning to predict stock prices based on historical data. The model uses a Random Forest Regressor to forecast the next closing price of a given stock. The data is fetched from Yahoo Finance and preprocessed to generate relevant features for the model.

## Features

- **Data Fetching**: Retrieves historical stock data at 5-minute intervals for the past month using `yfinance`.
- **Data Preprocessing**: Includes moving averages, volatility, and other relevant features.
- **Model Training**: Trains a Random Forest Regressor to predict the stock's next closing price.
- **Prediction**: Predicts the next closing price and visualizes it alongside historical data.
- **Visualization**: Plots actual closing prices and predicted next closing price on a graph.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/aman-kumar02/stock-price-prediction.git
2. **Navigate to the Project Directory**:
   cd stock-price-prediction
3. **Set Up Virtual Environment (optional but recommended)**:
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
4. **Install Required Packages**:
   Create a requirements.txt file or manually install the required libraries:
   pip install pandas yfinance scikit-learn matplotlib seaborn
   
**Usage**
Run the Script:
Execute the main script to fetch data, preprocess it, train the model, make predictions, and plot the results:
python stock_prediction.py

The script will print the Mean Squared Error of the model and the predicted next closing price. 
It will also display a plot showing the actual closing prices and the predicted next closing price.

**Modify Ticker Symbol**:

Open the stock_prediction.py file.
Change the ticker variable to any valid stock ticker symbol.

**Functions**
get_stock_data(ticker): Fetches historical stock data from Yahoo Finance.
preprocess_data(df): Prepares data by adding moving averages, volatility, and feature scaling.
train_model(X, y): Trains a Random Forest Regressor model.
predict_next_close(model, scaler, latest_data): Predicts the next closing price using the trained model.
plot_stock_prices(df, prediction, next_interval_time): Plots historical closing prices and the predicted next closing price.
Example Output
