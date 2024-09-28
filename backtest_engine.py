from datetime import datetime, timedelta
from data_manager import DataManager
from portfolio_manager import PortfolioManager
from signal_generator import SignalGenerator
from indicators import Indicators
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from shared_state import get_shared_state
from backtest_analyzer import BacktestAnalyzer
import matplotlib.pyplot as plt

class BacktestEngine:
    def __init__(self, initial_cash, min_cash, max_open_positions, start_date, end_date, params):
        self.start_date = start_date
        self.end_date = end_date
        self.params = params
        self.data_manager = DataManager()
        self.portfolio_manager = PortfolioManager(initial_cash, min_cash, max_open_positions, 
                                                  params['transaction_cost'], params['slippage'], 
                                                  self.data_manager)
        self.signal_generator = SignalGenerator(self.data_manager, Indicators(), self.portfolio_manager, params)
        self.PORTFOLIO = []
        self.shared_state = get_shared_state()
        self.BENCHMARK = []
        self.strategy = self.params['strategy']
        self.analyzer = BacktestAnalyzer()

    def run_backtest(self):
        current_date = self.start_date
        last_valid_asset_universe = None
        liquidation_orders = []

        self.analyzer.start_real_time_chart()

        while current_date < self.end_date:
            new_asset_universe = self.data_manager.update_investment_universe(current_date, 
                                                                              self.params['max_MC_rank'], 
                                                                              self.params['min_volume_thres'])
            if new_asset_universe is not None:
                last_valid_asset_universe = new_asset_universe

            ASSET_UNIVERSE = new_asset_universe if new_asset_universe is not None else last_valid_asset_universe

            if liquidation_orders:
                self.portfolio_manager.liquidate_position(current_date, liquidation_orders)

            self.portfolio_manager.process_orders(current_date)

            if self.params['rebalancing']:
                self.portfolio_manager.process_rebalancing(current_date)
                self.portfolio_manager.check_rebalancing(current_date)
           
            bull_market = self.signal_generator.regime_trend(current_date)

            if self.strategy == 'momentum':
                _, momentum_coins = self.signal_generator.rank_coins(current_date, ASSET_UNIVERSE, ranking_method='quality_momentum')
                if self.params["regime_filter"]:
                    if bull_market:
                        if momentum_coins is not None and len(momentum_coins) > 0:
                            self.signal_generator.momentum_entry_signals(current_date, momentum_coins, self.portfolio_manager.POSITIONS)
                
                else:
                    if momentum_coins is not None and len(momentum_coins) > 0:
                        self.signal_generator.momentum_entry_signals(current_date, momentum_coins, self.portfolio_manager.POSITIONS)
                
                if len(self.portfolio_manager.POSITIONS) > 0:
                    if self.params["regime_filter"]:
                        if not bull_market:
                            print('Exiting all long positions.')
                            self.portfolio_manager.exit_all_positions(position_type='long')
                        else:
                            self.signal_generator.momentum_exit_signals(current_date, momentum_coins, self.portfolio_manager.POSITIONS)
                    else:
                        self.signal_generator.momentum_exit_signals(current_date, momentum_coins, self.portfolio_manager.POSITIONS)
            
            elif self.strategy == 'mean_reversion':
                prices, reversion_coins = self.signal_generator.rank_coins(current_date, ASSET_UNIVERSE, ranking_method='mean_deviation')
                if not bull_market:
                    self.signal_generator.mean_reversion_entry_signals(current_date, prices, reversion_coins, self.portfolio_manager.POSITIONS, 
                                                                       accepted_signals='short_only')
                else:
                    self.signal_generator.mean_reversion_entry_signals(current_date, prices, reversion_coins, self.portfolio_manager.POSITIONS, 
                                                                       accepted_signals='long_only')
                
                if len(self.portfolio_manager.POSITIONS) > 0:
                    self.signal_generator.mean_reversion_exit_signals(current_date, self.portfolio_manager.POSITIONS)
                    if bull_market:
                        print('Exiting all short positions.')
                        self.portfolio_manager.exit_all_positions(position_type='short')
                    else:
                        print('Exiting all long positions.')
                        self.portfolio_manager.exit_all_positions(position_type='long')

            elif self.strategy == 'combined':
                _, momentum_coins = self.signal_generator.rank_coins(current_date, ASSET_UNIVERSE, ranking_method='quality_momentum')
                prices, reversion_coins = self.signal_generator.rank_coins(current_date, ASSET_UNIVERSE, ranking_method='mean_deviation')
                if bull_market:
                    self.signal_generator.momentum_entry_signals(current_date, momentum_coins, self.portfolio_manager.POSITIONS)
                else:
                    self.signal_generator.mean_reversion_entry_signals(current_date, prices, reversion_coins, self.portfolio_manager.POSITIONS)
                
                momentum_positions = {coin: pos for coin, pos in self.portfolio_manager.POSITIONS.items() if pos['strategy'] == 'momentum'}
                mean_reversion_positions = {coin: pos for coin, pos in self.portfolio_manager.POSITIONS.items() if pos['strategy'] == 'mean_reversion'}
                if len(momentum_positions) > 0:
                    if not bull_market:
                        print('Exiting all long positions.')
                        self.portfolio_manager.exit_all_positions(position_type='long')
                    else:
                        self.signal_generator.momentum_exit_signals(current_date, momentum_positions)

                if len(mean_reversion_positions) > 0:
                    self.signal_generator.mean_reversion_exit_signals(current_date, mean_reversion_positions)
                    if bull_market:
                        print('Exiting all short positions.')
                        self.portfolio_manager.exit_all_positions(position_type='short')


            self.portfolio_manager.update_portfolio(current_date)
            self.PORTFOLIO.append((current_date, self.portfolio_manager.AUM))

            # Update benchmark (BTC-USD) value
            btc_price = self.data_manager.fetch_current_price('BTCUSDT', current_date, 'Close')
            self.BENCHMARK.append((current_date, btc_price))

            self.analyzer.update_data(current_date, self.portfolio_manager.AUM, btc_price)

            liquidation_orders = self.portfolio_manager.check_stop_loss(current_date)

            current_date += timedelta(days=1)

            # Allow time for the plot to update
            plt.pause(0.003)

        # Generate backtest report
        self.generate_backtest_report()

        return self.PORTFOLIO

    def generate_backtest_report(self):
        output_dir = 'backtest_results'
        self.analyzer.generate_report(output_dir)

        print(f"Backtest completed. Results saved in {output_dir}")
