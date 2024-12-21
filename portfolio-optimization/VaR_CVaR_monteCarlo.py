#Monte carlo sim with value at riska dn conditional value at risk--australian markets
#Portfolio Optimization

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import datetime as dt 
import yfinance as yf

# Importing data
def get_data(stocks, start, end): 
    stockData = yf.download(stocks, start=start, end=end)['Close']
    returns = stockData.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return meanReturns, covMatrix

# Setting up stock list and date range
stockList = ['CBA', 'BHP', 'TLS', 'NAB', 'WBC', 'STO']
stocks = [stock + '.AX' for stock in stockList]
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=300)

# Fetching data
meanReturns, covMatrix = get_data(stocks, startDate, endDate)

# Generate random weights
weights = np.random.random(len(meanReturns))
weights /= np.sum(weights)

# Monte Carlo Simulation parameters
mc_sims = 100
T = 100  # timeframe (days)

# Prepare mean returns matrix
meanM = np.full(shape=(T, len(weights)), fill_value=meanReturns)
meanM = meanM.T

# Initialize portfolio simulation array
portfolio_sims = np.full(shape=(T, mc_sims), fill_value=0.0)
initialPortfolio = 10000

# Run Monte Carlo simulation
for m in range(0, mc_sims): 
    Z = np.random.normal(size=(T, len(weights)))
    L = np.linalg.cholesky(covMatrix)
    dailyReturns = meanM + np.inner(L, Z) 
    portfolio_sims[:, m] = np.cumprod(np.inner(weights, dailyReturns.T) + 1) * initialPortfolio

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(portfolio_sims)
plt.ylabel('Portfolio Value ($)')
plt.xlabel('Days')
plt.title('Monte Carlo Simulation of Stock Portfolio')
plt.tight_layout()
plt.show()


# Print some statistics
final_values = portfolio_sims[-1, :]
print(f"Mean final portfolio value: ${final_values.mean():.2f}")
print(f"Median final portfolio value: ${np.median(final_values):.2f}")
print(f"Min final portfolio value: ${final_values.min():.2f}")
print(f"Max final portfolio value: ${final_values.max():.2f}")


def mcVaR(returns, alpha=5): 
    """ Input: pandas series of returns
        Output: percentile on return distribution to a given confidence level alpha
    """

    if isinstance(returns, pd.Series): 
        return np.percentile(returns, alpha)
    else:
        raise TypeError("Expected a pandas data series")
    
def mcCVaR(returns, alpha=5): 
    """ Input: pandas series of returns
        Output: CVaR or Expected Shortfall to a given confidence level alpha
    """

    if isinstance(returns, pd.Series):
        belowVar = returns <= mcVaR(returns, alpha = alpha) 
        return returns[belowVar].mean()
    else:
        raise TypeError("Expected a pandas data series")
    
portResults = pd.Series(portfolio_sims[-1,:])

VaR = initialPortfolio - mcVaR(portResults, alpha=5)
CVaR = initialPortfolio -  mcCVaR(portResults, alpha=5)

print('VaR ${}'.format(round(VaR,2)))
print('CVaR ${}'.format(round(CVaR,2)))