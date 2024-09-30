#backtest_analyser.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from metrics import PerformanceMetrics
import seaborn as sns
import os
import pandas as pd
from matplotlib import dates as mdates

class BacktestAnalyzer:
    def __init__(self, risk_free_rate=0.0):
        self.portfolio_values = []
        self.benchmark_values = []
        self.dates = []
        self.risk_free_rate = risk_free_rate
        
        # Set up the initial plot
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        self.ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        self.portfolio_line, = self.ax.plot([], [], label='Portfolio')
        self.benchmark_line, = self.ax.plot([], [], label='Benchmark (BTC-USD)')
        self.ax.set_title('Momentum Portfolio vs Benchmark (%)')
        self.ax.set_xlabel('Date')
        self.ax.set_ylabel('Cumulative Return (%)')
        self.ax.legend()
        self.ax.grid(True)


    def update_data(self, date, portfolio_value, benchmark_value):
        if not isinstance(date, pd.Timestamp):
            date = pd.to_datetime(date)
        self.dates.append(date)
        self.portfolio_values.append(portfolio_value)
        self.benchmark_values.append(benchmark_value)

    def update_chart(self, frame):
        if not self.portfolio_values or not self.benchmark_values:
            return self.portfolio_line, self.benchmark_line

        portfolio_cumulative_return = 100 * (np.array(self.portfolio_values) / self.portfolio_values[0] - 1)
        benchmark_cumulative_return = 100 * (np.array(self.benchmark_values) / self.benchmark_values[0] - 1)

        self.portfolio_line.set_data(mdates.date2num(self.dates), portfolio_cumulative_return)
        self.benchmark_line.set_data(mdates.date2num(self.dates), benchmark_cumulative_return)

        self.ax.relim()
        self.ax.autoscale_view()

        return self.portfolio_line, self.benchmark_line

    def start_real_time_chart(self, interval=100):
        self.animation = FuncAnimation(self.fig, self.update_chart, interval=interval, blit=True, cache_frame_data=False)
        plt.gcf().autofmt_xdate()
        plt.show(block=False)
        plt.pause(0.1)  # Small pause to allow the window to open

    def save_final_chart(self, save_path):
        self.fig.savefig(save_path)
        plt.close(self.fig)

    def calculate_metrics(self):
        return PerformanceMetrics.calculate_metrics(self.portfolio, self.benchmark, self.risk_free_rate)

    def plot_portfolio_value(self, save_path):
        portfolio_series = pd.Series(self.portfolio_values, index=self.dates)
        benchmark_series = pd.Series(self.benchmark_values, index=self.dates)

        portfolio_cumulative_return = 100 * (portfolio_series / portfolio_series.iloc[0] - 1)
        benchmark_cumulative_return = 100 * (benchmark_series / benchmark_series.iloc[0] - 1)

        plt.figure(figsize=(12, 6))
        plt.plot(self.dates, portfolio_cumulative_return, label='Portfolio')
        plt.plot(self.dates, benchmark_cumulative_return, label='Benchmark (BTC-USD)')
        plt.title('Momentum Portfolio vs Benchmark (%) ')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return (%)')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()

    def plot_drawdown(self, save_path):
        portfolio_series = pd.Series(self.portfolio_values, index=self.dates)
        drawdown = (portfolio_series / portfolio_series.cummax()) - 1
        plt.figure(figsize=(12, 6))
        plt.plot(self.dates, drawdown)
        plt.title('Portfolio Drawdown')
        plt.xlabel('Date')
        plt.ylabel('Drawdown')
        plt.fill_between(self.dates, drawdown, 0, alpha=0.3)
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()

    def plot_rolling_sharpe(self, save_path, window=252):
        portfolio_series = pd.Series(self.portfolio_values, index=self.dates)
        returns = portfolio_series.pct_change().dropna()
        rolling_sharpe = returns.rolling(window).apply(
            lambda x: PerformanceMetrics.sharpe_ratio(x, self.risk_free_rate)
        )
        plt.figure(figsize=(12, 6))
        plt.plot(rolling_sharpe.index, rolling_sharpe.values)
        plt.title(f'Rolling Sharpe Ratio (Window: {window} days)')
        plt.xlabel('Date')
        plt.ylabel('Sharpe Ratio')
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()

    def plot_monthly_returns_heatmap(self, save_path):
        portfolio_series = pd.Series(self.portfolio_values, index=self.dates)
        returns = portfolio_series.pct_change().dropna()
        monthly_returns = returns.groupby([returns.index.year, returns.index.month]).sum().unstack()
        plt.figure(figsize=(12, 8))
        sns.heatmap(monthly_returns, annot=True, fmt='.2%', cmap='RdYlGn')
        plt.title('Monthly Returns Heatmap')
        plt.savefig(save_path)
        plt.close()

    def calculate_metrics(self):
        portfolio_series = pd.Series(self.portfolio_values, index=self.dates)
        benchmark_series = pd.Series(self.benchmark_values, index=self.dates)
        return PerformanceMetrics.calculate_metrics(portfolio_series, benchmark_series, self.risk_free_rate)

    def generate_report(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Generate plots
        self.plot_portfolio_value(os.path.join(output_dir, 'portfolio_value.png'))
        self.plot_drawdown(os.path.join(output_dir, 'drawdown.png'))
        self.plot_rolling_sharpe(save_path=os.path.join(output_dir, 'rolling_sharpe.png'))
        self.plot_monthly_returns_heatmap(os.path.join(output_dir, 'monthly_returns_heatmap.png'))

        # Calculate metrics
        metrics = self.calculate_metrics()

        # Save metrics
        for key, df in metrics.items():
            df.to_csv(os.path.join(output_dir, f'{key.lower().replace(" ", "_")}_metrics.csv'))

        print(f"Backtest analysis report generated in {output_dir}")