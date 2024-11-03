import numpy as np
from scipy.stats import norm
import QuantLib as ql

def vanilla_option(S, K, T, r, sigma, option='call'):
    """
    S: spot price
    K: strike price
    T: time to maturity
    r: risk-free interest rate
    sigma: standard deviation of price of underlying asset
    """
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = (np.log(S/K) + (r - 0.5*sigma**2)*T)/(sigma * np.sqrt(T))
    if option == 'call':
        p = (S*norm.cdf(d1, 0.0, 1.0) - K*np.exp(-r*T)*norm.cdf(d2, 0.0, 1.0))
    elif option == 'put':
        p = (K*np.exp(-r*T)*norm.cdf(-d2, 0.0, 1.0) - S*norm.cdf(-d1, 0.0, 1.0))
    else:
        return None
    return p

vanilla_option(50, 100, 1, 0.05, 0.25, option='call')

vanilla_option(50, 100, 1, 0.05, 0.25, option='put')


