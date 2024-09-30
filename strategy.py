import yaml
from rule_parser import RuleParser

class Strategy:
    def __init__(self, name, data_manager):
        self.name = name
        self.rule_parser = RuleParser(data_manager)
        self.load_rules()

    def load_rules(self):
        with open('strategies.yaml', 'r') as file:
            strategies = yaml.safe_load(file)['strategies']
        strategy_config = strategies[self.name]
        self.order_types = strategy_config['order_types']
        self.entry_rules = strategy_config['entry_rules']
        self.exit_rules = strategy_config['exit_rules']

    def generate_entry_signals(self, date, eligible_coins, positions):
        signals = {}
        for coin in eligible_coins:
            if coin not in positions:
                if 'long' in self.order_types and self.evaluate_rules(self.entry_rules, date, coin, 'long'):
                    print(f'Long entry signal detected for {coin} on {date}')
                    signals[coin] = {'signal': 1, 'strategy': self.name}
                elif 'short' in self.order_types and self.evaluate_rules(self.entry_rules, date, coin, 'short'):
                    print(f'Short entry signal detected for {coin} on {date}')
                    signals[coin] = {'signal': -1, 'strategy': self.name}
        return signals

    def generate_exit_signals(self, date, positions):
        signals = {}
        for coin, position in positions.items():
            position_type = 'long' if position['units'] > 0 else 'short'
            if self.evaluate_rules(self.exit_rules, date, coin, position_type):
                print(f'{position_type} exit signal detected for {coin} on {date}')
                signals[coin] = {'signal': -position['units'], 'strategy': self.name}
        return signals

    def evaluate_rules(self, rules, date, coin, order_type):
        return any(self.rule_parser.parse_rule(rule, date, coin, order_type) for rule in rules)

    def get_name(self):
        return self.name