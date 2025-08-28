import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import yahooquery
from yahooquery import search
from Portfolio import PortfolioOptimizer as po

# -----------------------
# Utility functions
# -----------------------

def get_ticker(company_name):
    """Convert company name → ticker (filter for NSE/BSE for India if available)."""
    result = search(company_name)
    try:
        equities = [item for item in result['quotes'] if item.get('quoteType') == 'EQUITY']
        
        # Prefer NSE (.NS) or BSE (.BO) listings
        for eq in equities:
            if eq['symbol'].endswith(('.NS')):
                return eq['symbol']
        
        # fallback: return first equity
        return equities[0]['symbol']
    except (KeyError, IndexError):
        raise ValueError(f"Could not find a valid ticker for {company_name}")


def get_tickers(company_names):
    """Takes a list of company names → returns list of tickers."""
    tickers = []
    for name in company_names:
        try:
            ticker = get_ticker(name)
            tickers.append(ticker)
        except ValueError as e:
            print(e)
            tickers.append(None)  # keep list length consistent
    return tickers








def get_data(tickers, period="5y"):
    
    data = yf.download(tickers, period=period).loc[:,"Close"]
    return data

def get_returns(price_data):
    return price_data.pct_change().dropna()

# -----------------------
# Streamlit UI
# -----------------------
st.title("📊 Portfolio Optimizer")

tickers_input = st.text_input("Enter stock tickers (comma-separated):")
capital = st.number_input("Enter total capital to allocate (₹):")
# period = st.selectbox("Select historical period:", ["1y", "3y", "5y", "10y"], index=2)
# freq = st.radio("Frequency scaling:", [252, 365], index=0)
# rf = st.number_input("Risk-free rate (annual):", value=0.055, step=0.001)

if st.button("Optimize Portfolio"):
    company_names=tickers_input.split(",")
    company_name=get_tickers(company_names)
    print(company_name)
    # tickers = [t.strip() for t in company_name.split(",")]
    try:
        opt = po(tickers=company_name)
        # price_data = get_data(tickers, period)
        returns = opt.compute_returns()

        # mean_returns = returns.mean().values
        # cov_matrix = returns.cov().values

        # result = optimize_portfolio(mean_returns, cov_matrix, rf, freq=freq)
        # weights = result.x
        print(opt.get_results())
      
        # ret, vol, sharpe = opt.get_results()#portfolio_perf(weights, mean_returns, cov_matrix, rf, freq=freq)
        results = opt.get_results()

        weights = results["weights"]
        ret = float(results["return"])
        vol = float(results["volatility"])
        sharpe = float(results["sharpe"])

        st.subheader("Optimal Portfolio Allocation")
        for t in  weights.items():
            st.write(f"**{t[0]}** → {t[1]:.2%}")
        # st.write(f"t: **→ {w}**")
        closing_price=opt.fetch_data()
        # closing_price = {}
        # for ticker in company_name:
        #     closing_price[ticker] = price_data.loc[:,ticker][-1]
        # return self.price_data, self.closing_price

        
        
        allocations = []
        for t in weights.items():#i, ticker in enumerate(tickers):
            weight = t[1]
            price = closing_price[t[0]]
            alloc_capital = weight * capital
            n_shares = alloc_capital // price   # floor division to avoid fractions
            
            allocations.append({
                "Company": t[0],
                "Weight": f"{t[1]:.2%}",
                "Current Market Price (₹)": round(price, 2),
                "Allocated Capital (₹)": round(alloc_capital, 2),
                "Shares": int(n_shares)
            })
            
        df_alloc = pd.DataFrame(allocations)
        st.dataframe(df_alloc)




        
        st.subheader("Performance Metrics")
        
        st.write(f"Expected Annual Return: **{ret:.2%}**")
        st.write(f"Expected Annual Volatility: **{vol:.2%}**")
        st.write(f"Sharpe Ratio: **{sharpe:.2f}**")

        

        best_weights, ret, vol, sharpe, fig = opt.simulate_and_plot_frontier()

        st.pyplot(fig)
        # st.write("Optimal Weights:", best_weights)
        # st.write(f"Return={ret:.4f}, Volatility={vol:.4f}, Sharpe={sharpe:.4f}")

    except Exception as e:
        st.error(f"Error fetching data: {e}")



if st.button("Minimum Variance Portfolio"):
    company_names=tickers_input.split(",")
    company_name=get_tickers(company_names)
    print(company_name)
    # tickers = [t.strip() for t in company_name.split(",")]
    try:
        opt = po(tickers=company_name)
        # price_data = get_data(tickers, period)
        returns = opt.compute_returns()

        # mean_returns = returns.mean().values
        # cov_matrix = returns.cov().values

        # result = optimize_portfolio(mean_returns, cov_matrix, rf, freq=freq)
        # weights = result.x
        print(opt.min_results())
      
        # ret, vol, sharpe = opt.get_results()#portfolio_perf(weights, mean_returns, cov_matrix, rf, freq=freq)
        results = opt.min_results()

        weights = results["weights"]
        ret = float(results["return"])
        vol = float(results["volatility"])
        sharpe = float(results["sharpe"])

        st.subheader("Minimum Variance Portfolio Allocation")
        for t in  weights.items():
            st.write(f"**{t[0]}** → {t[1]:.2%}")
        # st.write(f"t: **→ {w}**")

        st.subheader("Performance Metrics")
        
        st.write(f"Expected Annual Return: **{ret:.2%}**")
        st.write(f"Expected Annual Volatility: **{vol:.2%}**")
        st.write(f"Sharpe Ratio: **{sharpe:.2f}**")

    except Exception as e:
        st.error(f"Error fetching data: {e}")

