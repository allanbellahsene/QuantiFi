from datetime import datetime, timedelta
from data_manager import DataManager
from portfolio_manager import PortfolioManager
from signal_generator import SignalGenerator
from indicators import Indicators
import pandas as pd
from shared_state import get_shared_state
from backtest_analyzer import BacktestAnalyzer
import matplotlib.pyplot as plt
from strategy_factory import StrategyFactory
from config_loader import CONFIG
from metrics import PerformanceMetrics

class BacktestEngine:
    def __init__(self, initial_cash, min_cash, max_open_positions, start_date, end_date, params, strategy_name):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.params = params
        self.data_manager = DataManager()
        self.portfolio_manager = PortfolioManager(initial_cash, min_cash, max_open_positions, 
                                                  params['transaction_cost'], params['slippage'], 
                                                  self.data_manager)
        self.strategy = StrategyFactory.get_strategy(strategy_name, self.data_manager)
        self.signal_generator = SignalGenerator(self.data_manager, self.strategy)
        self.PORTFOLIO = []
        self.shared_state = get_shared_state()
        self.BENCHMARK = []
        self.analyzer = BacktestAnalyzer()
        self.performance_metrics = PerformanceMetrics()


    def run_backtest(self):
        current_date = self.start_date
        last_valid_asset_universe = None

        self.analyzer.start_real_time_chart()

        while current_date < self.end_date:
            new_asset_universe = self.data_manager.update_investment_universe(current_date, 
                                                                              self.params['max_MC_rank'], 
                                                                              self.params['min_volume_thres'])
            if new_asset_universe is not None:
                last_valid_asset_universe = new_asset_universe

            ASSET_UNIVERSE = new_asset_universe if new_asset_universe is not None else last_valid_asset_universe

            prices, eligible_coins = self.signal_generator.rank_coins(current_date, ASSET_UNIVERSE, self.params['ranking_method'])

            # Process orders based on the prioritized signals
            self.portfolio_manager.process_orders(current_date)

            if self.params['rebalancing']:
                self.portfolio_manager.process_rebalancing(current_date)
                self.portfolio_manager.check_rebalancing(current_date)

            # Generate and prioritize signals, including regime-based signals
            self.signal_generator.generate_and_prioritize_signals(current_date, eligible_coins, self.portfolio_manager.POSITIONS)


            self.portfolio_manager.update_portfolio(current_date)
            self.PORTFOLIO.append((current_date, self.portfolio_manager.AUM))

            # Update benchmark (BTC-USD) value
            btc_price = self.data_manager.fetch_current_price('BTCUSDT', current_date, 'Close')
            self.BENCHMARK.append((current_date, btc_price))

            self.analyzer.update_data(current_date, self.portfolio_manager.AUM, btc_price)

            current_date += timedelta(days=1)

            # Allow time for the plot to update
            plt.pause(0.001)

        # Generate backtest report
        self.generate_backtest_report()

        return self.PORTFOLIO

    def generate_backtest_report(self):
        output_dir = 'backtest_results'
        self.analyzer.generate_report(output_dir)
        
        # Convert PORTFOLIO and BENCHMARK to DataFrames
        portfolio_df = pd.DataFrame(self.PORTFOLIO, columns=['Date', 'Value'])
        portfolio_df.set_index('Date', inplace=True)
        benchmark_df = pd.DataFrame(self.BENCHMARK, columns=['Date', 'Value'])
        benchmark_df.set_index('Date', inplace=True)

        # Get trades from portfolio_manager
        trades = self.portfolio_manager.get_trades()

        # Calculate metrics
        #metrics = PerformanceMetrics.calculate_metrics(portfolio_df['Value'], benchmark_df['Value'], trades)

        # Ensure trades is a list of dictionaries
        if not isinstance(trades, list) or not all(isinstance(trade, dict) for trade in trades):
            print("Warning: Trades is not a list of dictionaries. Attempting to convert...")
            if isinstance(trades, pd.DataFrame):
                trades = trades.to_dict('records')
            else:
                raise ValueError("Unable to process trades data. Please check the format.")

        # Generate plots
        #self.performance_metrics.plot_equity_curve(portfolio_df['Value'], benchmark_df['Value'])
        #self.performance_metrics.plot_drawdown(portfolio_df['Value'].pct_change())
        #self.performance_metrics.plot_monthly_returns_heatmap(portfolio_df['Value'].pct_change())
        #self.performance_metrics.plot_trade_distribution(trades)
        #self.performance_metrics.plot_cumulative_returns(portfolio_df['Value'].pct_change(), benchmark_df['Value'].pct_change())
        #self.performance_metrics.plot_rolling_sharpe(portfolio_df['Value'].pct_change())
    

        self.performance_metrics.run_full_analysis(
            portfolio_df['Value'], 
            benchmark_df['Value'], 
            trades, 
            self.data_manager,  # Pass the DataManager instance
            current_date,       # Pass the current date
            risk_free_rate=0.02, 
            periods_per_year=365
        )

        # Print summary statistics
        print("\nBacktest Summary:")
        print(f"Strategy: {self.strategy.get_name()}")
        print(f"Start Date: {self.start_date.strftime('%Y-%m-%d')}")
        print(f"End Date: {self.end_date.strftime('%Y-%m-%d')}")
        print(f"Initial Portfolio Value: ${self.PORTFOLIO[0][1]:,.2f}")
        print(f"Final Portfolio Value: ${self.PORTFOLIO[-1][1]:,.2f}")
        
        total_return = (self.PORTFOLIO[-1][1] / self.PORTFOLIO[0][1] - 1) * 100
        print(f"Total Return: {total_return:.2f}%")
        
        # Calculate annualized return
        days = (self.end_date - self.start_date).days
        annualized_return = ((1 + total_return / 100) ** (365 / days) - 1) * 100
        print(f"Annualized Return: {annualized_return:.2f}%")
        
        print(f"\nDetailed results saved in {output_dir}")

if __name__ == "__main__":
    
    backtest = BacktestEngine(
        CONFIG['backtest']['initial_cash'],
        CONFIG['backtest']['min_cash'],
        CONFIG['backtest']['max_open_positions'],
        CONFIG['backtest']['start_date'],
        CONFIG['backtest']['end_date'],
        CONFIG['params'],
        CONFIG['strategy']
    )
    
    results = backtest.run_backtest()
