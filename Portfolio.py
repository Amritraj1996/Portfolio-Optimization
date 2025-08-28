import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
rf=5.5/36500
class PortfolioOptimizer:
    def __init__(self, tickers, period="5y", rf=rf, freq=365):
        """
        tickers : list of stock symbols
        period  : data period, default 5 years
        rf      : annual risk-free rate
        
        """
        self.tickers = tickers
        self.period = period
        self.rf = rf
        self.freq = freq
        self.price_data = None
        self.returns = None
        self.mean_returns = None
        self.cov_matrix = None
        self.opt_result = None

    # ----------------- Data Fetching -----------------
    def fetch_data(self):
        self.price_data = yf.download(
            self.tickers, period=self.period).loc[:,"Close"]
        self.closing_price = {}
        for ticker in self.tickers:
           
            self.closing_price[ticker] = self.price_data.loc[:,ticker][-1]
        return  self.closing_price #self.price_data,

    def compute_returns(self):
        if self.price_data is None:
            self.fetch_data()
        # self.returns = self.price_data.pct_change().dropna()
        # self.mean_returns = self.returns.mean().values
        # self.cov_matrix = self.returns.cov().values
        # return self.returns
        # self.price_data=df
        self.returns_df=pd.DataFrame()
        df=self.price_data
        for i in range(self.price_data.shape[1]):
            returns=[]
            for j in range(self.price_data.shape[0]-1):
                k=df.iloc[j+1,i]/df.iloc[j,i]-1
                returns.append(k)
            self.returns_df[df.columns[i]]=returns

        self.mean_returns=self.returns_df.mean()
        self.cov_matrix=self.returns_df.cov()
        
        return self.returns_df

       

    # ----------------- Portfolio Math -----------------
    def portfolio_perf(self, weights):
        port_return = np.dot(weights, self.mean_returns) 
        port_vol = np.sqrt(weights.T @ self.cov_matrix @ weights) 
        sharpe = (port_return - self.rf) / port_vol
        return port_return, port_vol, sharpe

    def neg_sharpe(self, weights):
        return -self.portfolio_perf(weights)[2]

    def min_volatility(self,weights):
        return -self.portfolio_perf(weights)[1]

    # ----------------- Optimization -----------------
    def optimize(self):
        n_assets = len(self.mean_returns)
        init_guess = np.repeat(1/n_assets, n_assets)
        bounds = tuple((0, 1) for _ in range(n_assets))
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

        self.opt_result = minimize(
            self.neg_sharpe, init_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,options={'maxiter': 1000, 'ftol': 1e-12, 'disp': True}
        )
     



        
        return self.opt_result


    def minimum_variance(self):
        n_assets = len(self.mean_returns)
        init_guess = np.repeat(1/n_assets, n_assets)
        bounds = tuple((0, 1) for _ in range(n_assets))
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

        self.min_result = minimize(
            self.min_volatility, init_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,options={'maxiter': 1000, 'ftol': 1e-12, 'disp': True}
        )
        return self.min_result

    # ----------------- Results -----------------
    def get_results(self):
        if self.opt_result is None:
            self.optimize()
        weights = self.opt_result.x
        ret, vol, sharpe = self.portfolio_perf(weights)
        print("hjfh",vol)
        return {
            "weights": dict(zip(self.tickers, weights)),
            "return": ret,
            "volatility": vol,
            "sharpe": sharpe
        }

    def min_results(self):
        if self.opt_result is None:
            self.minimum_variance()
        weights = self.min_result.x
        ret, vol, sharpe = self.portfolio_perf(weights)
        return {
            "weights": dict(zip(self.tickers, weights)),
            "return": ret,
            "volatility": vol,
            "sharpe": sharpe
        }


  
    def simulate_and_plot_frontier(self,n_sim=5000):
        """
        Simulates portfolios, plots efficient frontier, 
        and returns best portfolio weights + Sharpe ratio.
        """
       
        n_assets = len(self.mean_returns)
        results = np.zeros((3, n_sim))
        weights_record = []
       
        for i in range(n_sim):
            # Random weights
            weights = np.random.random(n_assets)
            weights /= np.sum(weights)
    
            port_return = np.dot(weights, self.mean_returns)
            port_vol = np.sqrt(weights.T @ self.cov_matrix @ weights)
            sharpe = (port_return - rf) / port_vol
    
            results[0, i] = port_return
            results[1, i] = port_vol
            results[2, i] = sharpe
            weights_record.append(weights)

        # max_sharpe_idx = np.argmax(results[2])


    
        # Find max Sharpe
        max_sharpe_idx = np.argmax(results[2])
        best_weights = weights_record[max_sharpe_idx]
        best_return = results[0, max_sharpe_idx]
        best_vol = results[1, max_sharpe_idx]
        best_sharpe = results[2, max_sharpe_idx]
        sdp_max, rp_max = results[1, max_sharpe_idx], results[0, max_sharpe_idx]
        # Plot efficient frontier
        plt.figure(figsize=(10, 6))
        plt.scatter(results[1, :], results[0, :], c=results[2, :], cmap='viridis', alpha=0.6)
        plt.colorbar(label="Sharpe Ratio")
        plt.scatter(best_vol, best_return, c="red", marker="*", s=200, label="Max Sharpe")

        x = np.linspace(0, max(results[1, :]), 100)
        slope = (rp_max - rf) / sdp_max
        cml = rf + slope * x
        plt.plot(x, cml, linestyle="--", color="black", label="Capital Market Line")
                
        plt.xlabel("Volatility")
        plt.ylabel("Return")
        plt.title("Efficient Frontier")
        plt.legend()
        plt.grid(True)
    
        return best_weights, best_return, best_vol, best_sharpe, plt
