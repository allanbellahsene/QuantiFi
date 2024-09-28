import numpy as np
import pandas as pd

class PerformanceMetrics:
    @staticmethod
    def total_return(series):
        return (series.iloc[-1] / series.iloc[0]) - 1

    @staticmethod
    def annual_return(returns, periods_per_year=252):
        total_return = (1 + returns).prod()
        n_periods = len(returns)
        return total_return ** (periods_per_year / n_periods) - 1

    @staticmethod
    def annual_volatility(returns, periods_per_year=252):
        return returns.std() * np.sqrt(periods_per_year)

    @staticmethod
    def sharpe_ratio(returns, risk_free_rate=0.02, periods_per_year=252):
        excess_returns = returns - risk_free_rate / periods_per_year
        return np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()

    @staticmethod
    def max_drawdown(returns):
        cum_returns = (1 + returns).cumprod()
        peak = cum_returns.expanding(min_periods=1).max()
        drawdown = (cum_returns / peak) - 1
        return drawdown.min()

    @staticmethod
    def calmar_ratio(returns, periods_per_year=252):
        annual_ret = PerformanceMetrics.annual_return(returns, periods_per_year)
        max_dd = PerformanceMetrics.max_drawdown(returns)
        return -annual_ret / max_dd if max_dd != 0 else np.inf

    @staticmethod
    def sortino_ratio(returns, risk_free_rate=0.02, periods_per_year=252):
        excess_returns = returns - risk_free_rate / periods_per_year
        downside_returns = excess_returns[excess_returns < 0]
        downside_deviation = np.sqrt(np.mean(downside_returns**2)) * np.sqrt(periods_per_year)
        return (np.mean(excess_returns) * periods_per_year) / downside_deviation if downside_deviation != 0 else np.inf

    @staticmethod
    def beta(returns, market_returns):
        covariance = np.cov(returns, market_returns)[0][1]
        market_variance = np.var(market_returns)
        return covariance / market_variance if market_variance != 0 else np.nan

    @staticmethod
    def alpha(returns, market_returns, risk_free_rate=0.02, periods_per_year=252):
        beta = PerformanceMetrics.beta(returns, market_returns)
        ann_return = PerformanceMetrics.annual_return(returns, periods_per_year)
        ann_market_return = PerformanceMetrics.annual_return(market_returns, periods_per_year)
        return ann_return - (risk_free_rate + beta * (ann_market_return - risk_free_rate))

    @staticmethod
    def calculate_metrics(portfolio_values, benchmark_values, risk_free_rate=0.02, periods_per_year=252):
        portfolio_returns = portfolio_values.pct_change().dropna()
        benchmark_returns = benchmark_values.pct_change().dropna()

        metrics = {
            'Total Return': PerformanceMetrics.total_return(portfolio_values),
            'Annual Return': PerformanceMetrics.annual_return(portfolio_returns, periods_per_year),
            'Annual Volatility': PerformanceMetrics.annual_volatility(portfolio_returns, periods_per_year),
            'Sharpe Ratio': PerformanceMetrics.sharpe_ratio(portfolio_returns, risk_free_rate, periods_per_year),
            'Max Drawdown': PerformanceMetrics.max_drawdown(portfolio_returns),
            'Calmar Ratio': PerformanceMetrics.calmar_ratio(portfolio_returns, periods_per_year),
            'Sortino Ratio': PerformanceMetrics.sortino_ratio(portfolio_returns, risk_free_rate, periods_per_year),
            'Beta': PerformanceMetrics.beta(portfolio_returns, benchmark_returns),
            'Alpha': PerformanceMetrics.alpha(portfolio_returns, benchmark_returns, risk_free_rate, periods_per_year)
        }

        return pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])