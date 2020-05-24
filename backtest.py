import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import namedtuple, defaultdict
from scipy.stats import norm

_riskfree = 0.00

def _d1(S, K, T, sigma, r=.0, q=.0):
    with np.errstate(divide='ignore'):
        return (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))

def _d2(S, K, T, sigma, r=.0, q=.0):    
    return _d1(S, K, T, sigma, r, q) - sigma*np.sqrt(T)

def call(S, K, T, sigma, r=.0, q=.0):
    if T <= 0: return max(0, S-K)
    d1, d2 = _d1(S, K, T, sigma, r, q), _d2(S, K, T, sigma, r, q)
    call = S * np.exp(-q*T) * norm.cdf(d1, 0.0, 1.0)  -  K * np.exp(-r*T) * norm.cdf(d2, 0.0, 1.0)
    return call

def put(S, K, T, sigma, r=.0, q=.0):
    if T<=0: return max(0, K-S)
    d1, d2 = _d1(S, K, T, sigma, r, q), _d2(S, K, T, sigma, r, q)
    put = K * np.exp(-r*T) * norm.cdf(-d2, 0.0, 1.0)  -  S * np.exp(-q*T) * norm.cdf(-d1, 0.0, 1.0)
    return put

def Stock(ticker=None):
    Class = namedtuple('Stock', ('ticker', ))
    return Class(ticker.upper())

def Option(right=None, underlying=None, strike=None, lastTradeDate=None, multiplier=100):
    Class = namedtuple('Option', ('right', 'underlying', 'strike', 'lastTradeDate', 'multiplier', ))
    return Class(
        right.upper(),
        underlying.upper(),
        float(round(strike, 2)),
        pd.to_datetime(lastTradeDate),        
        int(multiplier),
    )

def get_close(ticker):
    df = pd.read_csv(f'./prices/{ticker}.csv', index_col=0, parse_dates=True)
    close = df['Adj Close']
    return close

class Account:
    def __init__(self, name, ini_fund, stock_prices, vix):
        self._name = name
        self._ini_fund = ini_fund
        self._cash = 0
        self._stock_pos = defaultdict(int)
        self._option_pos = defaultdict(int)
        self._stock_prices = stock_prices
        self._vix = vix
        self._dashboard = pd.DataFrame(columns=('Cash', 'Stock', 'Option', 'NAV'), dtype=float)

    # market

    def stock_price_at(self, at, contract):
        return self._stock_prices[contract.ticker].loc[at]

    def option_price_at(self, at, contract):
        pricing = call if contract.right.upper()=='CALL' else put
        price = pricing(
            S=self._stock_prices[contract.underlying].loc[at],
            K=contract.strike,
            T=(contract.lastTradeDate - at).days / 365,
            sigma=self._vix.loc[at]/100,
            r=_riskfree,
        )
        return price

    # order
    def deposit(self, amount):
        self._cash += amount

    def trade_stock(self, at, contract, share):
        price = self._stock_prices[contract.ticker].loc[at]
        self._cash -= price*share
        self._stock_pos[contract] += share

    def trade_stock_target_percentage(self, at, contract, lv):
        price = self._stock_prices[contract.ticker].loc[at]
        target_share = int(self.net_asset_value(at)[-1]*lv/price)
        current_share = self._stock_pos[contract]
        net_share = target_share - current_share
        self.trade_stock(at, contract, net_share)

    def close_all_stock_position(self, at):
        for contract,pos in self._stock_pos.items():
            self.trade_stock(at, contract, -pos)

    def trade_option(self, at, contract, share):
        price = self.option_price_at(at, contract)
        self._cash -= price*share*contract.multiplier
        self._option_pos[contract] += share

    def close_all_option_positions(self, at):
        for contract,share in self._option_pos.items():
            amount = self._option_pos[contract]
            self.trade_option(at, contract, -amount)

    # settlement
    def net_asset_value(self, at):
        cash_val = self._cash
        stock_val = sum(self.stock_price_at(at, contract)*share
                        for contract,share in self._stock_pos.items())
        option_val = sum(self.option_price_at(at, contract)*share*contract.multiplier
                        for contract,share in self._option_pos.items())
        nav = cash_val + stock_val + option_val
        return cash_val, stock_val, option_val, nav

    def settlement(self, at):
        for s in tuple(self._stock_pos.keys()):
            if self._stock_pos[s] == 0:
                del self._stock_pos[s]
        for o in tuple(self._option_pos.keys()):
            if self._option_pos[o] == 0:
                del self._option_pos[o]
        vals = self.net_asset_value(at)
        self._dashboard.loc[at] = vals
        return vals

class Strategy:
    def _set_args(self, kwargs):
        if not hasattr(self, '_args'): self._args = {}
        for key,val in kwargs.items(): 
            setattr(self, f'_{key}', val)
            self._args = {**self._args, **kwargs}

    def __init__(self, **kwargs):
        self._set_args(kwargs)
        self._stock_prices = {self._stock_ticker:get_close(self._stock_ticker), }
        self._vix = get_close('^VIX')
        self._acc = Account('Active', self._ini_fund, self._stock_prices, self._vix)
        self._acc_bm = Account('Benchmark', self._ini_fund, self._stock_prices, self._vix)

    def _iron_condor(self, acc, today, last, ttm):
        pxA = round(last * (1 - 2 * self._put_offset), 2)
        pxB = round(last * (1 - 1 * self._put_offset), 2)
        pxC = round(last * (1 + 1 * self._call_offset), 2)
        pxD = round(last * (1 + 2 * self._call_offset), 2)
        ltd = str((today + pd.DateOffset(days=ttm)).date())
        qty = int(acc.net_asset_value(today)[-1]*self._target_lv/last/100)

        acc.close_all_option_positions(today)
        acc.trade_option(today, Option('put', self._stock_ticker, pxA, ltd), +1 * qty)
        acc.trade_option(today, Option('put', self._stock_ticker, pxB, ltd), -1 * qty)
        acc.trade_option(today, Option('call', self._stock_ticker, pxC, ltd), -1 * qty)
        acc.trade_option(today, Option('call', self._stock_ticker, pxD, ltd), +1 * qty)

    def run(self, **kwargs):
        self._set_args(kwargs)
        timeline = self._stock_prices[self._stock_ticker].loc[self._start:].index        
        for i,today in enumerate(timeline):
            # daily
            last = self._stock_prices[self._stock_ticker].loc[today]
            # initialize account
            if i==0:
                for acc in (self._acc, self._acc_bm):
                    acc.deposit(self._ini_fund)
            # option strategy
            if i%self._rebal_freq==0:
                self._iron_condor(self._acc, today, last, self._short_ttm)
                self._iron_condor(self._acc_bm, today, last, self._long_ttm)


            # at every day end
            print(f'{i:5d} | {today.date()} ', end='')
            for acc in (self._acc, self._acc_bm):
                nav = acc.settlement(today)[-1]
                print(f' | {acc._name}: {nav:12,.2f}', end='')
            print(end='\t\t\r')
        return self

    def evaluate(self, **kwargs):
        self._set_args(kwargs)
        df = pd.concat((self._acc._dashboard['NAV'], self._acc_bm._dashboard['NAV']), axis=1)
        df.columns=('Strategy', 'Benchmark', )
        fig, ax = plt.subplots(2, 1, figsize=(16, 8), sharex=False, gridspec_kw={'height_ratios': (3, 1,)})
        # performance chart
        title = ', '.join((f'{k}={v}' for k, v in self._args.items()))
        for name, ts in df.iteritems():
            def metrics(name, ts):
                def cal_sharpe(ts, rf=0.025):
                    lndiffs = np.log(ts).diff()
                    mu = lndiffs.mean() * 255
                    sigma = lndiffs.std() * 252 ** .5
                    sharpe = (mu - rf) / sigma
                    return mu, sigma, sharpe

                def cal_drawdown(ts):
                    ts = np.log(ts)
                    run_max = np.maximum.accumulate(ts)
                    end = (run_max - ts).idxmax()
                    start = (ts.loc[:end]).idxmax()
                    low = ts.at[end]
                    high = ts.at[start]
                    dd = np.exp(low) / np.exp(high) - 1
                    pts = {'high': start, 'low': end}
                    duration = len(ts.loc[start:end])
                    return dd, pts, duration

                mu, sigma, sharpe = cal_sharpe(ts)
                dd, pts, duration = cal_drawdown(ts)
                text = (f'\n{name} |mu:{mu:.2%} | sigma:{sigma:.2%} | sharpe:{sharpe:.2%} | '
                        f'drawdown:{dd:.2%} ({pts["high"].date()}-{pts["low"].date()}, {duration}d)')
                return text

            title += metrics(name, ts)            
            label = f'{name} | Ending value: {ts[-1]:,.0f}, Total return: {ts[-1]/ts[0]-1:,.2%}'
            ax[0].plot(ts, label=label)
            ax[0].legend(loc='upper left')
        ax[0].set_title(title)
        # ratio chart
        ratio = self._acc._dashboard['Option'] / self._acc._dashboard['NAV']
        ax[1].plot(ratio)
        ax[1].set_title('Option value as percentage of NAV')
        plt.show()
        return self

if __name__ == "__main__":
    Strategy(
        stock_ticker='SPY',     # underlying stock to long
        ini_fund=1e6,           # initial fund
        short_ttm=10,           # option days to maturity, in normal day
        long_ttm=15,            # a longer time to maturity as control experiment
        rebal_freq=5,           # how often to roll over to new contract, in business day
        target_lv=1.0,          # ratio of the underlying assets size and NAV
        call_offset=.02,        # difference between last price and put strike price
        put_offset=.02,         # difference between last price and put strike price
    ).run(
        start='1995-01-01',     # backtest start date
    ).evaluate()