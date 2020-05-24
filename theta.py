import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


def _d1(S, K, T, sigma, r=.0, q=.0):
    return (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))

def _d2(S, K, T, sigma, r=.0, q=.0):
    return _d1(S, K, T, sigma, r, q) - sigma*np.sqrt(T)

def call_theta(S, K, T, sigma, r=.0, q=.0, yearD=365):
    d1, d2 = _d1(S, K, T, sigma, r, q), _d2(S, K, T, sigma, r, q)
    call_theta = 1/yearD * ( \
        -(S*sigma*np.exp(-q*T)/(2*np.sqrt(T)) * 1/(np.sqrt(2*np.pi)) * np.exp(-d1**2/2)) \
        -r*K*np.exp(-r*T)*norm.cdf(d2) \
        +q*S*np.exp(-q*T)*norm.cdf(d1) )
    return call_theta

def put_theta(S, K, T, sigma, r=.0, q=.0, yearD=365):
    d1, d2 = _d1(S, K, T, sigma, r, q), _d2(S, K, T, sigma, r, q)
    call_theta = 1/yearD * ( \
        -(S*sigma*np.exp(-q*T)/(2*np.sqrt(T)) * 1/(np.sqrt(2*np.pi)) * np.exp(-d1**2/2)) \
        +r*K*np.exp(-r*T)*norm.cdf(-d2) \
        -q*S*np.exp(-q*T)*norm.cdf(-d1) )
    return call_theta


class App:
    def __init__(self, S, sigma, T, underlying, multiplier, year=365):
        self.S = S
        self.sigma = sigma
        self.T = T
        self.underlying = underlying
        self.multiplier = multiplier
        self.year = year

    def portfolio_theta(self, portfolio, t):
        ptheta = 0
        for legs in portfolio:
            pos, right, strike, T = legs.split(' ')
            pos, right, strike, T  = int(pos), right.upper(), float(strike), int(T)
            theta_func = {'CALL': call_theta, 'PUT': put_theta}[right]
            ptheta += pos*theta_func(self.S, strike, (T-t)/self.year, self.sigma)*self.multiplier
        return ptheta

    def run(self, name, portfolio):
        x_t = np.arange(self.T-1, 0, -1)
        y_theta = self.portfolio_theta(portfolio, x_t[::-1])
        fig, ax = plt.subplots(1,1, figsize=(8,4))
        fig.suptitle(f'Theta of {name} versus time')
        ax.invert_xaxis()
        ax.plot(x_t, y_theta, c='b')
        plt.show()
        return self


if __name__ == '__main__':
    App(
        S = 300,
        sigma = 0.2,
        T = 90,
        underlying='SPY',
        multiplier=100,
    ).run(
        name='European Call',
        portfolio=(
            '+1 CALL 300 90',
        ),
    ).run(
        name='European Put',
        portfolio=(
            '+1 PUT 300 90',
        ),
    ).run(
        name='Iron Condor',
        portfolio=(
            '+1 PUT 290 90',
            '-1 PUT 295 90',
            '-1 CALL 305 90',
            '+1 CALL 310 90',
        )
    )
