import operator
import pandas as pd

class RuleParser:
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.ops = {
            '>': operator.gt,
            '<': operator.lt,
            '>=': operator.ge,
            '<=': operator.le,
            '==': operator.eq,
            '!=': operator.ne
        }
    
    def parse_rule(self, rule, date, coin, order_type):
        #print(f"Parsing rule: {rule} for {coin} on {date}")  # Debug print
        if 'and' in rule:
            conditions = rule.split(' and ')
            return all(self.parse_rule(condition, date, coin, order_type) for condition in conditions)
        
        if f"'{order_type}'" in rule:
            return True
        
        parts = rule.split()
        if len(parts) < 3:
            print(f"Invalid rule format: {rule}")
            return False
        left = self.parse_expression(parts[0], date, coin)
        op = self.ops[parts[1]]
        if op is None:
            print(f"Invalid operator: {parts[1]}")
            return False

        right = self.parse_expression(parts[2], date, coin)
        return op(left, right)

    def parse_expression(self, expr, date, coin):
        #print(f"Parsing expression: {expr}")  # Debug print
        if expr.replace('.', '').isdigit():
            return float(expr)
        elif '(' in expr:
            func_name, args_str = expr.split('(', 1)
            args_str = args_str.rstrip(')')
            args = [arg.strip() for arg in args_str.split(',')]
            #print(f"Function call: {func_name} with args {args}")  # Debug print
            if func_name == 'SMA':
                if len(args) != 2:
                    print(f"Error: SMA requires 2 arguments, got {len(args)}")
                    return None
                return self.SMA(date, coin, args[0], args[1])
            return getattr(self, func_name)(date, coin, *args)
        elif expr in ["'long'", "'short'"]:
            return expr.strip("'")
        else:
            return self.get_price_data(date, coin, expr)

    def get_price_data(self, date, coin, column='Close'):
        return self.data_manager.fetch_current_price(coin, date, column)

    def HighestHigh(self, date, coin, column, period):
        lookback_date = date - pd.Timedelta(days=int(period))
        prices = self.data_manager.fetch_historical_prices([coin], lookback_date, date, price_col=column)
        return prices.max()

    def SMA(self, date, coin, column, period):
        #print(f"SMA called with date: {date}, coin: {coin}, column: {column}, period: {period}")  # Debug print
        try:
            period = int(period)
            if period <= 0:
                raise ValueError("SMA period must be a positive integer")
        except ValueError as e:
            #print(f"Error in SMA calculation: {e}")
            #print(f"Date: {date}, Coin: {coin}, Column: {column}, Period: {period}")
            return None

        lookback_date = date - pd.Timedelta(days=period)
        prices = self.data_manager.fetch_historical_prices([coin], lookback_date, date, price_col=column)
        if prices.empty:
            print(f"No price data for {coin} from {lookback_date} to {date}")
            return None
        return prices.rolling(window=period).mean().iloc[-1]
