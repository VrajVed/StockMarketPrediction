from neuralprophet import NeuralProphet
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt


stock_symbol = input("Enter your stock symbol (ex. RELIANCE.NS): ") #Stock Company Name [Example RELIANCE]
start_date = "2010-01-01"
end_date = "2024-01-01"

stock_data = yf.download(stock_symbol, start = start_date, end = end_date)

print(stock_data.head())
stock_data.to_csv("stock_data.csv")

stocks = pd.read_csv('stock_data.csv')
stocks['Date'] = pd.to_datetime(stocks['Date'])
stocks = stocks[['Date', 'Close']]
stocks.columns = ['ds','y']

plt.plot(stocks['ds'], stocks['y'], label = 'actual', c = 'g')

#TRAINING THE MODEL 

model = NeuralProphet()
model.fit(stocks)

future = model.make_future_dataframe(stocks, periods = 365)

forecast = model.predict(future)
actual_prediction = model.predict(stocks)

plt.plot(actual_prediction['ds'], actual_prediction['yhat1'], label = "prediction_Actual", c = 'r')
plt.plot(forecast['ds'], forecast['yhat1'], label = 'future_prediction', c = 'b')
plt.plot(stocks['ds'], stocks['y'], label = 'actual', c = 'g')
plt.legend()
plt.title(stock_symbol)
plt.show()
