import datetime

import yfinance as yf
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import griddata

from qiskit_finance.circuit.library import LogNormalDistribution
from qiskit_finance.applications.estimation import EuropeanCallPricing
from qiskit_algorithms import IterativeAmplitudeEstimation
from qiskit.primitives import StatevectorSampler

ticker = yf.Ticker("MSFT")
history = ticker.history(period="1y")

if history.empty or len(history) < 2:
    raise ValueError(f"Failed to load enough data for ticker {ticker.ticker}")

CurrentPrice = float(history["Close"].iloc[-1])

log_returns = np.log(history["Close"]/history["Close"].shift(1)).dropna()
daily_volatility = log_returns.std()
sigma = float(daily_volatility * np.sqrt(252))

days_to_strike_price = 40
Part_of_the_year = days_to_strike_price / 365
strike_price = CurrentPrice
r = 0.05 #Risk-free interest rate

#Monte Carlo
possible_outputs = 100_000
np.random.seed(67)

K = np.random.standard_normal(possible_outputs)
FuturePrice = CurrentPrice * np.exp((r - sigma**2*0.5)*Part_of_the_year + sigma*np.sqrt(Part_of_the_year)*K)
payoff = np.maximum(FuturePrice - strike_price,0)
discounting_factor = np.exp(-r * Part_of_the_year)
discounting_payoffs = discounting_factor * payoff
avg_payoffs = discounting_payoffs.mean()
SEM = discounting_payoffs.std()/np.sqrt(possible_outputs) #Standard Error of the Mean

#Quantum version
num_qubit = 5
qubit_future_price = (r - sigma**2*0.5)*Part_of_the_year+np.log(CurrentPrice)
avg_qubit_future_price = qubit_future_price.mean()
partial_volatility = sigma * np.sqrt(Part_of_the_year)
dispersion = (np.exp(partial_volatility**2) - 1) * np.exp(2*qubit_future_price + partial_volatility**2)
future_price_dispersion = np.sqrt(dispersion)
mean_future_price = np.exp(qubit_future_price + partial_volatility**2 / 2)

low_limit = max(0.01, mean_future_price - 3*future_price_dispersion)
high_limit = mean_future_price + 3*future_price_dispersion

FuturePriceSpread = LogNormalDistribution(num_qubits = num_qubit, mu = qubit_future_price, sigma = partial_volatility**2, bounds = (low_limit, high_limit))
european_call = EuropeanCallPricing(num_qubit, strike_price, rescaling_factor = 0.1, bounds = (low_limit, high_limit), uncertainty_model = FuturePriceSpread)
format_european_call = european_call.to_estimation_problem()

quant_sim = StatevectorSampler(seed=67, default_shots = 4000)
quant_usage = IterativeAmplitudeEstimation(epsilon_target=0.01, alpha=0.05, sampler=quant_sim)
result = quant_usage.estimate(format_european_call)

quant_payoffs = european_call.interpret(result)
price_quant = quant_payoffs * discounting_factor

alg_low, alg_high = result.confidence_interval_processed
alg_price_quant = (alg_low * discounting_factor, alg_high * discounting_factor)

difference = abs(avg_payoffs - price_quant)
difference_percent = difference / avg_payoffs * 100

min_border = avg_payoffs - 1.96 * SEM
max_border = avg_payoffs + 1.96 * SEM

quant_low, quant_high = alg_price_quant

print("Ticker:", ticker.ticker)
print(f"Current stock price: ${CurrentPrice:,.4f}")
print(f"Strike price: ${strike_price:,.4f}")
print(f"Annualized volatility: {sigma * 100:,.2f}%")
print("Risk-free rate: ", r * 100, "%")
print("Monte Carlo")
print(f"Payoff: ${avg_payoffs:,.4f}")
print(f"Standard error: {SEM:,.4f}")
print(f"95% confidence interval: ${min_border:,.2f}, ${max_border:,.2f}")
print("Quantum Model")
print(f"Payoff: ${price_quant:,.4f}")
print(f"95% confidence interval: ${quant_low:,.2f}, ${quant_high:,.2f}")
print("Comparison of methods")
print(f"Absolute difference: ${difference:,.4f}")
print(f"Percentage difference: {difference_percent:,.2f}%")

# --- Real Implied Volatility Surface from Yahoo Finance option chains ---
# Filters are tuned to keep only liquid, near-the-money quotes, since deep
# in/out-of-the-money and low-volume contracts often carry unreliable
# implied volatility values that distort the surface.

expiration_dates = ticker.options  # all available expiration dates for MSFT options

if not expiration_dates:
    raise ValueError(f"No option expiration dates available for ticker {ticker.ticker}")

today = datetime.datetime.now()

# Keep only reasonably liquid, near-the-money quotes
MIN_VOLUME = 10
MIN_OPEN_INTEREST = 50
MONEYNESS_RANGE = (0.8, 1.2)
MAX_IV = 1.0  # 100% annualized IV as a sanity cutoff

moneyness_points = []
maturity_points = []
iv_points = []

for exp_date in expiration_dates:
    opt_chain = ticker.option_chain(exp_date)
    calls = opt_chain.calls

    if calls.empty:
        continue

    exp_datetime = datetime.datetime.strptime(exp_date, "%Y-%m-%d")
    days_to_expiry = (exp_datetime - today).days

    if days_to_expiry <= 0:
        continue

    calls = calls.copy()
    calls["moneyness"] = calls["strike"] / CurrentPrice
    calls["volume"] = calls["volume"].fillna(0)
    calls["openInterest"] = calls["openInterest"].fillna(0)

    valid_calls = calls[
        (calls["impliedVolatility"] > 0)
        & (calls["impliedVolatility"] <= MAX_IV)
        & (calls["volume"] >= MIN_VOLUME)
        & (calls["openInterest"] >= MIN_OPEN_INTEREST)
        & (calls["moneyness"] >= MONEYNESS_RANGE[0])
        & (calls["moneyness"] <= MONEYNESS_RANGE[1])
    ]

    for _, row in valid_calls.iterrows():
        moneyness_points.append(row["moneyness"])
        maturity_points.append(days_to_expiry)
        iv_points.append(row["impliedVolatility"])

if len(iv_points) < 10:
    raise ValueError(
        "Not enough liquid option quotes to build a reliable implied volatility surface. "
        "Try relaxing MIN_VOLUME, MIN_OPEN_INTEREST, or MONEYNESS_RANGE."
    )

moneyness_points = np.array(moneyness_points)
maturity_points = np.array(maturity_points)
iv_points = np.array(iv_points)

# Build a regular grid and interpolate the scattered real data onto it
grid_moneyness = np.linspace(moneyness_points.min(), moneyness_points.max(), 100)
grid_maturity = np.linspace(maturity_points.min(), maturity_points.max(), 50)
M, T = np.meshgrid(grid_moneyness, grid_maturity)

volatility_surface = griddata(
    points=(moneyness_points, maturity_points),
    values=iv_points,
    xi=(M, T),
    method="linear"
)

fig = plt.figure(figsize=(10, 7))
chart_volume = fig.add_subplot(111, projection='3d')
surface = chart_volume.plot_surface(M, T, volatility_surface, cmap=cm.turbo, linewidth=0, antialiased=True, alpha=1)
chart_volume.set_title(f"Implied Volatility Surface ({ticker.ticker}, liquid near-the-money quotes)", pad=10)
chart_volume.set_xlabel("Moneyness (Strike / Spot)", labelpad=10)
chart_volume.set_ylabel("Days to Maturity", labelpad=10)
chart_volume.set_zlabel("Implied Volatility", labelpad=10)
chart_volume.view_init(elev=30, azim=235)
fig.colorbar(surface, aspect=15, pad=0.1, label="Implied Volatility")
plt.tight_layout()
plt.savefig("Volatility_surface.png", dpi=200, bbox_inches='tight')
