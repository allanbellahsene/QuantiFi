#metrics.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

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
    def calculate_trade_returns(trades_df, current_prices=None):
        # Sort trades by date and instrument
        trades_df = trades_df.sort_values(['instrument', 'date'])
        
        # Initialize lists to store trade information
        trade_info = []
        
        # Group trades by instrument
        for instrument, group in trades_df.groupby('instrument'):
            open_trades = []
            
            for _, trade in group.iterrows():
                if trade['action'] == 'OPEN':
                    open_trades.append(trade)
                elif trade['action'] == 'CLOSE' and open_trades:
                    open_trade = open_trades.pop(0)
                    
                    # Calculate trade return for closed trades
                    entry_price = open_trade['price']
                    exit_price = trade['price']
                    units = open_trade['units']
                    
                    if units > 0:
                        order = 'LONG'
                        trade_return = (exit_price / entry_price - 1) * 100
                    else:
                        order = 'SHORT'
                        trade_return = (entry_price / exit_price - 1) * 100
                    
                    # Calculate holding time
                    entry_date = pd.to_datetime(open_trade['date'])
                    exit_date = pd.to_datetime(trade['date'])
                    holding_time = (exit_date - entry_date).days
                    
                    trade_info.append({
                        'Instrument': instrument,
                        'Order': order,
                        'Trade Return (%)': trade_return,
                        'Entry Price': entry_price,
                        'Exit Price': exit_price,
                        'Entry Date': entry_date,
                        'Exit Date': exit_date,
                        'Holding Time (days)': holding_time,
                        'Strategy': open_trade['strategy'],
                        'Status': 'Closed'
                    })
            
            # Handle open trades
            if open_trades and current_prices is not None:
                for open_trade in open_trades:
                    entry_price = open_trade['price']
                    current_price = current_prices.get(instrument)
                    if current_price is None:
                        continue
                    
                    units = open_trade['units']
                    
                    if units > 0:
                        order = 'LONG'
                        trade_return = (current_price / entry_price - 1) * 100
                    else:
                        order = 'SHORT'
                        trade_return = (entry_price / current_price - 1) * 100
                    
                    entry_date = pd.to_datetime(open_trade['date'])
                    holding_time = (pd.Timestamp.now() - entry_date).days
                    
                    trade_info.append({
                        'Instrument': instrument,
                        'Order': order,
                        'Trade Return (%)': trade_return,
                        'Entry Price': entry_price,
                        'Exit Price': current_price,
                        'Entry Date': entry_date,
                        'Exit Date': pd.Timestamp.now(),
                        'Holding Time (days)': holding_time,
                        'Strategy': open_trade['strategy'],
                        'Status': 'Open'
                    })
        
        # Create DataFrame from trade_info
        results_df = pd.DataFrame(trade_info)
        
        # Sort by Entry Date
        if not results_df.empty:
            results_df = results_df.sort_values('Entry Date')
        
        return results_df

    @staticmethod
    def analyze_trades(trades, current_prices=None):
        if not trades:
            return pd.DataFrame(columns=['Value'])

        trades_df = pd.DataFrame(trades)
        trades_df['date'] = pd.to_datetime(trades_df['date'])
        
        trades_df = PerformanceMetrics.calculate_trade_returns(trades_df, current_prices)
        print(trades_df)
        trades_df.to_csv('backtest_results/trades_df.csv')

        closed_trades = trades_df[trades_df['Status'] == 'Closed']
        open_trades = trades_df[trades_df['Status'] == 'Open']

        metrics = {
            'Total Trades': len(trades_df),
            'Closed Trades': len(closed_trades),
            'Open Trades': len(open_trades),
            'Winning Trades': sum(trades_df['Trade Return (%)'] > 0),
            'Losing Trades': sum(trades_df['Trade Return (%)'] <= 0),
            'Win Rate': sum(trades_df['Trade Return (%)'] > 0) / len(trades_df) if len(trades_df) > 0 else 0,
            'Average Trade Return': trades_df['Trade Return (%)'].mean(),
            'Average Winning Trade': trades_df.loc[trades_df['Trade Return (%)'] > 0, 'Trade Return (%)'].mean(),
            'Average Losing Trade': trades_df.loc[trades_df['Trade Return (%)'] <= 0, 'Trade Return (%)'].mean(),
            'Largest Winning Trade': trades_df['Trade Return (%)'].max(),
            'Largest Losing Trade': trades_df['Trade Return (%)'].min(),
            'Average Holding Period (days)': (trades_df['Holding Time (days)']).mean(),
        }

        return pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])

    def calculate_metrics(portfolio_values, benchmark_values, trades, risk_free_rate=0.0, periods_per_year=365, current_prices=None):
        portfolio_returns = portfolio_values.pct_change().dropna()
        benchmark_returns = benchmark_values.pct_change().dropna()

        overall_metrics = {
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

        trade_metrics = PerformanceMetrics.analyze_trades(trades, current_prices)
        trade_frequency = PerformanceMetrics.analyze_trade_frequency(trades)

        return {
            'Overall Metrics': pd.DataFrame.from_dict(overall_metrics, orient='index', columns=['Value']),
            'Trade Metrics': trade_metrics,
            'Trade Frequency': trade_frequency
        }


    @staticmethod
    def analyze_trade_frequency(trades):
        if not trades:
            return pd.Series()

        trades_df = pd.DataFrame(trades)

        if 'instrument' not in trades_df.columns:
            print("Warning: 'instrument' column not found in trades data")
            return pd.Series()


    @staticmethod
    def plot_equity_curve(portfolio_values, benchmark_values):
        plt.figure(figsize=(12, 6))
        plt.plot(portfolio_values.index, portfolio_values, label='Portfolio')
        plt.plot(benchmark_values.index, benchmark_values, label='Benchmark')
        plt.title('Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_drawdown(returns):
        cum_returns = (1 + returns).cumprod()
        peak = cum_returns.expanding(min_periods=1).max()
        drawdown = (cum_returns / peak) - 1
        
        plt.figure(figsize=(12, 6))
        plt.plot(drawdown.index, drawdown)
        plt.title('Drawdown')
        plt.xlabel('Date')
        plt.ylabel('Drawdown')
        plt.fill_between(drawdown.index, drawdown, 0, alpha=0.3)
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_monthly_returns_heatmap(returns):
        monthly_returns = returns.resample('M').agg(lambda x: (x + 1).prod() - 1)
        monthly_returns_df = monthly_returns.to_frame()
        monthly_returns_df['Year'] = monthly_returns_df.index.year
        monthly_returns_df['Month'] = monthly_returns_df.index.month
        monthly_returns_pivot = monthly_returns_df.pivot(index='Month', columns='Year', values='Value')
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(monthly_returns_pivot, annot=True, fmt='.2%', cmap='RdYlGn')
        plt.title('Monthly Returns Heatmap')
        plt.show()

    @staticmethod
    def plot_trade_returns_distribution(trades, current_prices=None):
        trades_df = pd.DataFrame(trades)
        trades_df['date'] = pd.to_datetime(trades_df['date'])
        
        trade_returns_df = PerformanceMetrics.calculate_trade_returns(trades_df, current_prices)
        
        plt.figure(figsize=(12, 6))
        
        # Create the histogram
        sns.histplot(data=trade_returns_df, x='Trade Return (%)', kde=True, hue='Status', multiple="stack")
        
        # Add a vertical line at 0
        plt.axvline(x=0, color='r', linestyle='--', label='Break-even')
        
        # Calculate and display mean and median
        mean_return = trade_returns_df['Trade Return (%)'].mean()
        median_return = trade_returns_df['Trade Return (%)'].median()
        
        plt.axvline(x=mean_return, color='g', linestyle='-', label=f'Mean: {mean_return:.2f}%')
        plt.axvline(x=median_return, color='b', linestyle='-', label=f'Median: {median_return:.2f}%')
        
        # Set labels and title
        plt.xlabel('Trade Return (%)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Trade Returns')
        
        # Add legend
        plt.legend()
        
        # Add summary statistics as text
        stats = f"Total Trades: {len(trade_returns_df)}\n"
        stats += f"Closed Trades: {sum(trade_returns_df['Status'] == 'Closed')}\n"
        stats += f"Open Trades: {sum(trade_returns_df['Status'] == 'Open')}\n"
        stats += f"Profitable Trades: {sum(trade_returns_df['Trade Return (%)'] > 0)} ({sum(trade_returns_df['Trade Return (%)'] > 0) / len(trade_returns_df) * 100:.2f}%)\n"
        stats += f"Loss-making Trades: {sum(trade_returns_df['Trade Return (%)'] < 0)} ({sum(trade_returns_df['Trade Return (%)'] < 0) / len(trade_returns_df) * 100:.2f}%)\n"
        stats += f"Max Profit: {trade_returns_df['Trade Return (%)'].max():.2f}%\n"
        stats += f"Max Loss: {trade_returns_df['Trade Return (%)'].min():.2f}%"
        
        plt.text(0.95, 0.95, stats, transform=plt.gca().transAxes, va='top', ha='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        
        # Show the plot
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_trade_frequency(trade_frequency):
        plt.figure(figsize=(12, 6))
        trade_frequency.plot(kind='bar')
        plt.title('Trade Frequency by Instrument')
        plt.xlabel('Instrument')
        plt.ylabel('Number of Trades')
        plt.xticks(rotation=45)
        plt.show()

    @staticmethod
    def plot_cumulative_returns(portfolio_returns, benchmark_returns):
        cum_portfolio_returns = (1 + portfolio_returns).cumprod()
        cum_benchmark_returns = (1 + benchmark_returns).cumprod()
        
        plt.figure(figsize=(12, 6))
        plt.plot(cum_portfolio_returns.index, cum_portfolio_returns, label='Portfolio')
        plt.plot(cum_benchmark_returns.index, cum_benchmark_returns, label='Benchmark')
        plt.title('Cumulative Returns')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_rolling_sharpe(returns, window=252):
        rolling_sharpe = returns.rolling(window=window).apply(lambda x: PerformanceMetrics.sharpe_ratio(x))
        
        plt.figure(figsize=(12, 6))
        plt.plot(rolling_sharpe.index, rolling_sharpe)
        plt.title(f'Rolling Sharpe Ratio (Window: {window} days)')
        plt.xlabel('Date')
        plt.ylabel('Sharpe Ratio')
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_exposure_over_time(trades, portfolio_values):
        if not trades:
            print("No trades to plot exposure.")
            return

        trades_df = pd.DataFrame(trades)
        
        if 'entry_date' not in trades_df.columns or 'exposure' not in trades_df.columns:
            print("Required columns 'entry_date' or 'exposure' not found in trades data")
            return

        daily_exposure = trades_df.set_index('entry_date')['exposure'].resample('D').sum()
        daily_exposure = daily_exposure.reindex(portfolio_values.index).fillna(0)
        
        plt.figure(figsize=(12, 6))
        plt.plot(daily_exposure.index, daily_exposure)
        plt.title('Daily Exposure Over Time')
        plt.xlabel('Date')
        plt.ylabel('Exposure')
        plt.grid(True)
        plt.show()

    @staticmethod
    def check_look_ahead_bias(trades, portfolio_values):
        if not trades:
            print("No trades to check for look-ahead bias.")
            return

        trades_df = pd.DataFrame(trades)

        if 'Entry_date' not in trades_df.columns or 'Exit_date' not in trades_df.columns:
            print("Required columns 'Entry_date' or 'Exit_date' not found in trades data")
            return

        for _, trade in trades_df.iterrows():
            entry_date = trade['Entry_date']
            exit_date = trade['Exit_date']
            
            if entry_date not in portfolio_values.index or exit_date not in portfolio_values.index:
                print(f"Warning: Trade dates not in portfolio values index. Entry: {entry_date}, Exit: {exit_date}")
            
            if exit_date <= entry_date:
                print(f"Warning: Exit date {exit_date} is not after entry date {entry_date}")

    @staticmethod
    def check_overfitting(in_sample_returns, out_of_sample_returns):
        in_sample_sharpe = PerformanceMetrics.sharpe_ratio(in_sample_returns)
        out_of_sample_sharpe = PerformanceMetrics.sharpe_ratio(out_of_sample_returns)
        
        print(f"In-sample Sharpe Ratio: {in_sample_sharpe:.2f}")
        print(f"Out-of-sample Sharpe Ratio: {out_of_sample_sharpe:.2f}")
        
        if out_of_sample_sharpe < 0.5 * in_sample_sharpe:
            print("Warning: Potential overfitting detected. Out-of-sample performance is significantly worse.")

    @staticmethod
    def save_results(output_dir, metrics):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save overall metrics
        if metrics['Overall Metrics'] is not None:
            overall_metrics_path = os.path.join(output_dir, 'overall_metrics.json')
            with open(overall_metrics_path, 'w') as f:
                json.dump(metrics['Overall Metrics'].to_dict(), f, indent=4)
        
        # Save trade metrics
        if metrics['Trade Metrics'] is not None:
            trade_metrics_path = os.path.join(output_dir, 'trade_metrics.json')
            with open(trade_metrics_path, 'w') as f:
                json.dump(metrics['Trade Metrics'].to_dict(), f, indent=4)
        
        # Save trade frequency
        if metrics['Trade Frequency'] is not None:
            trade_frequency_path = os.path.join(output_dir, 'trade_frequency.json')
            with open(trade_frequency_path, 'w') as f:
                json.dump(metrics['Trade Frequency'].to_dict(), f, indent=4)
        
        print(f"Results saved in {output_dir}")

    @staticmethod
    def run_full_analysis(portfolio_values, benchmark_values, trades, data_manager, risk_free_rate=0.02, periods_per_year=252):
        # Get the current prices for open trades
        current_prices = {}
        for trade in trades:
            if trade['action'] == 'OPEN':
                current_prices[trade['instrument']] = data_manager.fetch_current_price(trade['instrument'], current_date, 'Close')
        
        metrics = PerformanceMetrics.calculate_metrics(portfolio_values, benchmark_values, trades, risk_free_rate, periods_per_year, current_prices)
        PerformanceMetrics.save_results('backtest_results', metrics)

        print("METRICS")
        print(metrics)
        
        print("Overall Metrics:")
        print(metrics['Overall Metrics'])
        print("\nTrade Metrics:")
        print(metrics['Trade Metrics'])
        
        portfolio_returns = portfolio_values.pct_change().dropna()
        benchmark_returns = benchmark_values.pct_change().dropna()
        
        PerformanceMetrics.plot_equity_curve(portfolio_values, benchmark_values)
        PerformanceMetrics.plot_drawdown(portfolio_returns)
        PerformanceMetrics.plot_monthly_returns_heatmap(portfolio_returns)
        PerformanceMetrics.plot_trade_returns_distribution(trades, current_prices)
        #PerformanceMetrics.plot_trade_frequency(metrics['Trade Frequency'])
        #PerformanceMetrics.plot_cumulative_returns(portfolio_returns, benchmark_returns)
        #PerformanceMetrics.plot_rolling_sharpe(portfolio_returns)
        #PerformanceMetrics.plot_exposure_over_time(trades, portfolio_values)
        
        #PerformanceMetrics.check_look_ahead_bias(trades, portfolio_values)
        
        # Assuming the first 70% of the data is in-sample and the rest is out-of-sample
        #split_index = int(len(portfolio_returns) * 0.7)
        #in_sample_returns = portfolio_returns[:split_index]
        #out_of_sample_returns = portfolio_returns[split_index:]
        #PerformanceMetrics.check_overfitting(in_sample_returns, out_of_sample_returns)