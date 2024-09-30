from strategy import Strategy

class StrategyFactory:
    @staticmethod
    def get_strategy(strategy_name, data_manager):
        return Strategy(strategy_name, data_manager)