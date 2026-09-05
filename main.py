import yfinance as yf
import numpy as np

ticker = yf.Ticker("MSFT")
history = ticker.history(period="1y")
CurrentPrice = float(history["Close"].iloc[-1])
log_returns = np.log(history["Close"]/history["Close"].shift(1)).dropna()
daily_volatility = log_returns.std()
sigma = float(daily_volatility * np.sqrt(252))

