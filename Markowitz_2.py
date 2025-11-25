"""
Package Import
"""
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
import warnings
import argparse
import sys

"""
Project Setup
"""
warnings.simplefilter(action="ignore", category=FutureWarning)

assets = [
    "SPY",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]

# Initialize Bdf and df
Bdf = pd.DataFrame()
for asset in assets:
    raw = yf.download(asset, start="2012-01-01", end="2024-04-01", auto_adjust = False)
    Bdf[asset] = raw['Adj Close']

df = Bdf.loc["2019-01-01":"2024-04-01"]

"""
Strategy Creation

Create your own strategy, you can add parameter but please remain "price" and "exclude" unchanged
"""


class MyPortfolio:
    """
    NOTE: You can modify the initialization function
    """

    def __init__(self, price, exclude, lookback=50, gamma=0):
        self.price = price
        self.returns = price.pct_change().fillna(0)
        self.exclude = exclude
        self.lookback = lookback
        self.gamma = gamma

    def calculate_weights(self):
        # Get the assets by excluding the specified column
        assets = self.price.columns[self.price.columns != self.exclude]

        # Calculate the portfolio weights
        self.portfolio_weights = pd.DataFrame(
            index=self.price.index, columns=self.price.columns
        )

        """
        TODO: Complete Task 4 Below
        """
        n_assets = len(assets)
        assets_list = list(assets)
        
        # XLK-focused strategy with trend timing
        sma_period = 200
        rebalance_freq = 5
        
        # Core holding: XLK (tech), XLY (consumer discretionary)
        core_assets = ['XLK', 'XLY']
        # Defensive: XLP (staples), XLV (healthcare), XLU (utilities)
        defensive_assets = ['XLP', 'XLV', 'XLU']
        
        current_weights = np.ones(n_assets) / n_assets
        
        for i in range(self.lookback, len(self.price)):
            if (i - self.lookback) % rebalance_freq == 0:
                # Check XLK trend
                xlk_idx = assets_list.index('XLK') if 'XLK' in assets_list else 0
                xlk_prices = self.price['XLK'].iloc[max(0, i-sma_period):i+1]
                
                if len(xlk_prices) >= sma_period:
                    sma = xlk_prices.iloc[-sma_period:].mean()
                    current_price = xlk_prices.iloc[-1]
                    uptrend = current_price > sma
                else:
                    uptrend = True
                
                current_weights = np.zeros(n_assets)
                
                if uptrend:
                    # In uptrend: 60% XLK, 40% XLY
                    if 'XLK' in assets_list:
                        current_weights[assets_list.index('XLK')] = 0.6
                    if 'XLY' in assets_list:
                        current_weights[assets_list.index('XLY')] = 0.4
                else:
                    # In downtrend: split among defensive sectors
                    for asset in defensive_assets:
                        if asset in assets_list:
                            current_weights[assets_list.index(asset)] = 1.0 / len(defensive_assets)
                
                # Ensure weights sum to 1
                if current_weights.sum() > 0:
                    current_weights = current_weights / current_weights.sum()
                else:
                    current_weights = np.ones(n_assets) / n_assets
            
            for j, asset in enumerate(assets_list):
                self.portfolio_weights.loc[self.price.index[i], asset] = current_weights[j]
            self.portfolio_weights.loc[self.price.index[i], self.exclude] = 0.0

        
        """
        TODO: Complete Task 4 Above
        """

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = self.returns.copy()
        assets = self.price.columns[self.price.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        # Ensure portfolio returns are calculated
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns


if __name__ == "__main__":
    # Import grading system (protected file in GitHub Classroom)
    from grader_2 import AssignmentJudge
    
    parser = argparse.ArgumentParser(
        description="Introduction to Fintech Assignment 3 Part 12"
    )

    parser.add_argument(
        "--score",
        action="append",
        help="Score for assignment",
    )

    parser.add_argument(
        "--allocation",
        action="append",
        help="Allocation for asset",
    )

    parser.add_argument(
        "--performance",
        action="append",
        help="Performance for portfolio",
    )

    parser.add_argument(
        "--report", action="append", help="Report for evaluation metric"
    )

    parser.add_argument(
        "--cumulative", action="append", help="Cumulative product result"
    )

    args = parser.parse_args()

    judge = AssignmentJudge()
    
    # All grading logic is protected in grader_2.py
    judge.run_grading(args)
