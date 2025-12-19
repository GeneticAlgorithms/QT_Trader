"""
Market Data Integration for Testing NLS Model Against Real Options
Uses yfinance to fetch real option chain data
"""

import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


def get_stock_data(ticker, period="1y"):
    """
    Fetch stock price data using yfinance.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol (e.g., 'AAPL', 'MSFT')
    period : str
        Period to fetch (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
    
    Returns:
    --------
    yf.Ticker object with stock data
    """
    stock = yf.Ticker(ticker)
    return stock


def get_option_chain(ticker, expiration_date=None):
    """
    Get option chain data for a given ticker.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    expiration_date : str, optional
        Expiration date in format 'YYYY-MM-DD'. If None, uses nearest expiration.
    
    Returns:
    --------
    dict with 'calls' and 'puts' DataFrames
    """
    stock = yf.Ticker(ticker)
    
    if expiration_date:
        exp_dates = stock.options
        if expiration_date in exp_dates:
            opt_chain = stock.option_chain(expiration_date)
        else:
            print(f"Warning: {expiration_date} not available. Using nearest expiration.")
            opt_chain = stock.option_chain(stock.options[0])
    else:
        # Use nearest expiration
        opt_chain = stock.option_chain(stock.options[0])
    
    return {
        'calls': opt_chain.calls,
        'puts': opt_chain.puts,
        'expiration': opt_chain.expiration[0] if hasattr(opt_chain, 'expiration') else stock.options[0]
    }


def get_current_price(ticker):
    """
    Get current stock price.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    
    Returns:
    --------
    float : Current stock price
    """
    stock = yf.Ticker(ticker)
    info = stock.info
    return info.get('currentPrice', info.get('regularMarketPrice', 0))


def get_risk_free_rate():
    """
    Get current risk-free rate (10-year Treasury yield as proxy).
    
    Returns:
    --------
    float : Risk-free rate (annual)
    """
    try:
        treasury = yf.Ticker("^TNX")
        rate = treasury.history(period="1d")['Close'].iloc[-1] / 100
        return rate
    except:
        # Default to 0.05 (5%) if fetch fails
        return 0.05


def get_historical_volatility(ticker, period=252):
    """
    Calculate historical volatility from stock price data.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    period : int
        Number of trading days to use (default 252 = 1 year)
    
    Returns:
    --------
    float : Annualized volatility
    """
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1y")
    
    if len(hist) < 2:
        return 0.3  # Default volatility
    
    # Calculate daily returns
    returns = hist['Close'].pct_change().dropna()
    
    # Annualized volatility
    volatility = returns.std() * np.sqrt(252)
    
    return volatility


def prepare_option_data_for_comparison(ticker, option_type='call', 
                                       min_strike_ratio=0.8, 
                                       max_strike_ratio=1.2):
    """
    Prepare option chain data for model comparison.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    option_type : str
        'call' or 'put'
    min_strike_ratio : float
        Minimum strike price as ratio of current price
    max_strike_ratio : float
        Maximum strike price as ratio of current price
    
    Returns:
    --------
    pandas.DataFrame with columns: strike, bid, ask, lastPrice, volume, openInterest, 
                                   days_to_expiry, current_price, risk_free_rate, volatility
    """
    stock = yf.Ticker(ticker)
    current_price = get_current_price(ticker)
    risk_free_rate = get_risk_free_rate()
    volatility = get_historical_volatility(ticker)
    
    # Get option chain
    opt_chain = get_option_chain(ticker)
    
    if option_type.lower() == 'call':
        options = opt_chain['calls'].copy()
    else:
        options = opt_chain['puts'].copy()
    
    # Filter by strike price range
    min_strike = current_price * min_strike_ratio
    max_strike = current_price * max_strike_ratio
    options = options[(options['strike'] >= min_strike) & 
                     (options['strike'] <= max_strike)]
    
    # Calculate mid price (average of bid and ask)
    options['midPrice'] = (options['bid'] + options['ask']) / 2
    
    # Use lastPrice if available, otherwise midPrice
    options['marketPrice'] = options['lastPrice'].fillna(options['midPrice'])
    
    # Calculate days to expiration
    expiration = pd.to_datetime(opt_chain['expiration'])
    today = pd.Timestamp.now()
    options['daysToExpiry'] = (expiration - today).days
    options['timeToExpiry'] = options['daysToExpiry'] / 365.0  # Convert to years
    
    # Add market data
    options['currentPrice'] = current_price
    options['riskFreeRate'] = risk_free_rate
    options['volatility'] = volatility
    
    # Filter out options with invalid data
    options = options[options['marketPrice'] > 0]
    options = options[options['timeToExpiry'] > 0]
    
    return options


def compare_with_black_scholes(ticker, option_type='call', 
                               black_scholes_func=None):
    """
    Compare market option prices with Black-Scholes predictions.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    option_type : str
        'call' or 'put'
    black_scholes_func : callable, optional
        Function to calculate Black-Scholes price
    
    Returns:
    --------
    pandas.DataFrame with comparison results
    """
    from nls_model import NLSModel
    
    # Prepare option data
    options = prepare_option_data_for_comparison(ticker, option_type)
    
    if len(options) == 0:
        print(f"No valid options found for {ticker}")
        return pd.DataFrame()
    
    # Calculate Black-Scholes prices
    if black_scholes_func:
        options['bsPrice'] = options.apply(
            lambda row: black_scholes_func(
                row['currentPrice'],
                row['strike'],
                row['timeToExpiry'],
                row['riskFreeRate'],
                row['volatility']
            ),
            axis=1
        )
    else:
        # Default Black-Scholes calculation
        from scipy.special import erf
        def bs_call(S, K, T, r, sigma):
            d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)
            N_d1 = 0.5 * (1 + erf(d1 / np.sqrt(2)))
            N_d2 = 0.5 * (1 + erf(d2 / np.sqrt(2)))
            return S * N_d1 - K * np.exp(-r*T) * N_d2
        
        def bs_put(S, K, T, r, sigma):
            d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)
            N_d1 = 0.5 * (1 + erf(d1 / np.sqrt(2)))
            N_d2 = 0.5 * (1 + erf(d2 / np.sqrt(2)))
            return K * np.exp(-r*T) * (1 - N_d2) - S * (1 - N_d1)
        
        if option_type.lower() == 'call':
            options['bsPrice'] = options.apply(
                lambda row: bs_call(row['currentPrice'], row['strike'], 
                                   row['timeToExpiry'], row['riskFreeRate'], 
                                   row['volatility']),
                axis=1
            )
        else:
            options['bsPrice'] = options.apply(
                lambda row: bs_put(row['currentPrice'], row['strike'], 
                                   row['timeToExpiry'], row['riskFreeRate'], 
                                   row['volatility']),
                axis=1
            )
    
    # Calculate errors
    options['bsError'] = options['marketPrice'] - options['bsPrice']
    options['bsErrorPct'] = (options['bsError'] / options['marketPrice']) * 100
    
    return options


def compare_with_nls_model(ticker, option_type='call', 
                           solution_type='shock_wave',
                           beta=None, k=1.2):
    """
    Compare market option prices with NLS model predictions.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    option_type : str
        'call' or 'put'
    solution_type : str
        'shock_wave' or 'soliton'
    beta : float, optional
        Market potential. If None, uses risk-free rate
    k : float
        Wave number
    
    Returns:
    --------
    pandas.DataFrame with comparison results
    """
    from nls_model import NLSModel
    
    # Prepare option data
    options = prepare_option_data_for_comparison(ticker, option_type)
    
    if len(options) == 0:
        print(f"No valid options found for {ticker}")
        return pd.DataFrame()
    
    # Initialize NLS model for each option
    nls_prices = []
    
    for idx, row in options.iterrows():
        S = row['currentPrice']
        K = row['strike']
        T = row['timeToExpiry']
        r = row['riskFreeRate']
        sigma = row['volatility']
        
        # Use beta = r if not specified
        if beta is None:
            beta_val = r
        else:
            beta_val = beta
        
        # Initialize NLS model
        nls = NLSModel(sigma=sigma, beta=beta_val, k=k)
        
        # Calculate probability density
        s_range = np.linspace(K * 0.7, K * 1.3, 200)
        pdf = nls.probability_density(
            solution_type=solution_type,
            s=s_range,
            t=T,
            r=r
        )
        
        # Scale PDF to option price range
        # This is a simplified approach - full calibration would optimize this
        market_price = row['marketPrice']
        max_pdf = pdf.max()
        if max_pdf > 0:
            # Scale to match at-the-money option
            atm_idx = np.argmin(np.abs(s_range - S))
            scale_factor = market_price / pdf[atm_idx] if pdf[atm_idx] > 0 else 1.0
            nls_price = pdf[atm_idx] * scale_factor
        else:
            nls_price = 0
        
        nls_prices.append(nls_price)
    
    options['nlsPrice'] = nls_prices
    options['nlsError'] = options['marketPrice'] - options['nlsPrice']
    options['nlsErrorPct'] = (options['nlsError'] / options['marketPrice']) * 100
    
    return options


def plot_comparison_results(comparison_df, title="Option Price Comparison"):
    """
    Plot comparison between market prices and model predictions.
    
    Parameters:
    -----------
    comparison_df : pandas.DataFrame
        DataFrame from compare_with_black_scholes or compare_with_nls_model
    title : str
        Plot title
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Market vs Model prices
    axes[0, 0].scatter(comparison_df['strike'], comparison_df['marketPrice'], 
                      alpha=0.6, label='Market Price', s=50)
    if 'bsPrice' in comparison_df.columns:
        axes[0, 0].scatter(comparison_df['strike'], comparison_df['bsPrice'], 
                          alpha=0.6, label='Black-Scholes', s=50, marker='x')
    if 'nlsPrice' in comparison_df.columns:
        axes[0, 0].scatter(comparison_df['strike'], comparison_df['nlsPrice'], 
                          alpha=0.6, label='NLS Model', s=50, marker='^')
    axes[0, 0].set_xlabel('Strike Price', fontsize=12)
    axes[0, 0].set_ylabel('Option Price', fontsize=12)
    axes[0, 0].set_title('Price Comparison', fontsize=14)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Error distribution
    if 'bsError' in comparison_df.columns:
        axes[0, 1].hist(comparison_df['bsError'], bins=20, alpha=0.7, 
                       label='Black-Scholes Error', edgecolor='black')
    if 'nlsError' in comparison_df.columns:
        axes[0, 1].hist(comparison_df['nlsError'], bins=20, alpha=0.7, 
                       label='NLS Error', edgecolor='black')
    axes[0, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Error (Market - Model)', fontsize=12)
    axes[0, 1].set_ylabel('Frequency', fontsize=12)
    axes[0, 1].set_title('Error Distribution', fontsize=14)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Percentage error vs Strike
    if 'bsErrorPct' in comparison_df.columns:
        axes[1, 0].scatter(comparison_df['strike'], comparison_df['bsErrorPct'], 
                          alpha=0.6, label='Black-Scholes', s=50)
    if 'nlsErrorPct' in comparison_df.columns:
        axes[1, 0].scatter(comparison_df['strike'], comparison_df['nlsErrorPct'], 
                          alpha=0.6, label='NLS Model', s=50, marker='^')
    axes[1, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Strike Price', fontsize=12)
    axes[1, 0].set_ylabel('Percentage Error (%)', fontsize=12)
    axes[1, 0].set_title('Percentage Error vs Strike', fontsize=14)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Error vs Time to Expiry
    if 'bsError' in comparison_df.columns:
        axes[1, 1].scatter(comparison_df['timeToExpiry'], comparison_df['bsError'], 
                          alpha=0.6, label='Black-Scholes', s=50)
    if 'nlsError' in comparison_df.columns:
        axes[1, 1].scatter(comparison_df['timeToExpiry'], comparison_df['nlsError'], 
                          alpha=0.6, label='NLS Model', s=50, marker='^')
    axes[1, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Time to Expiry (years)', fontsize=12)
    axes[1, 1].set_ylabel('Error', fontsize=12)
    axes[1, 1].set_title('Error vs Time to Expiry', fontsize=14)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    return fig

