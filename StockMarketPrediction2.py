import datetime
import streamlit as st
from neuralprophet import NeuralProphet
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Stock Market Prediction", page_icon=":chart_with_upwards_trend:")
st.title("Stock Market Prediction Algorithm")

st.write("### Enter Data")
# SELECTION BOX

stock_symbol = st.selectbox("Select prefered stock to predict: ",('RELIANCE.NS', 'NESTLEIND.NS','HDFCLIFE.NS','BAJAJFINSV.NS',
'NTPC.NS','SHRIRAMFIN.NS', 'ULTRACEMCO.NS','TATACONSUM.NS','BAJFINANCE.NS','KOTAKBANK.NS','HEROMOTOCO.NS','BAJAJ-AUTO.NS',
'BRITANNIA.NS','APOLLOHOSP.NS','BHARTIARTL.NS','TATASTEEL.NS','WIPRO.NS','LT.NS','INDUSINDBK.NS','ITC.NS','TITAN.NS','COALINDIA.NS',
'ADANIENT.NS','MARUTI.NS','ONGC.NS','CIPLA.NS','TCS.NS','HINDALCO.NS','TRENT.NS'), index=None, placeholder="Select Stock Symbol...", args=None, kwargs=None,label_visibility="visible")

# OUTPUT
st.write("You selected: ", stock_symbol)

try:
    start_date = str(st.date_input("Enter Start Date", value=None))
    end_date = str(st.date_input("Enter End Date", value=None))

    stock_data = yf.download(stock_symbol, start = start_date, end = end_date)
    # PRINT DATASET   
    st.write(stock_data.head())
    stock_data.to_csv("stock_data.csv")

    stocks = pd.read_csv('stock_data.csv')
    stocks['Date'] = pd.to_datetime(stocks['Date'])
    stocks = stocks[['Date', 'Close']]
    stocks.columns = ['ds','y']

    x = stocks['ds']
    y = stocks['y']
    fig, ax = plt.subplots()
    ax.plot(x,y)
    plt.xlabel("Timeframe")
    plt.ylabel("Price in Rs")
    st.pyplot(fig)

    model = NeuralProphet()
    model.fit(stocks)

    future = model.make_future_dataframe(stocks, periods = 365)

    forecast = model.predict(future)
    actual_prediction = model.predict(stocks)

    fig, ax = plt.subplots()
    x1 = actual_prediction['ds']
    y1 = actual_prediction['yhat1']
    ax.plot(x1,y1)
    plt.xlabel("Timeframe")
    plt.ylabel("Price in Rs")
    st.pyplot(fig)

    fig, ax = plt.subplots()
    x2 = forecast['ds']
    y2 = forecast['yhat1']
    ax.plot(x2,y2)
    plt.xlabel("Timeframe")
    plt.ylabel("Price in Rs")
    st.pyplot(fig)

    st.pyplot(x1,y1)
    st.legend()
    st.title(stock_symbol)
    st.show()

except AttributeError as AE:
    st.exception(AE)
except ValueError as VE:
    st.error("Value Error")
