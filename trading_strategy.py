import numpy as np

class TradingStrategy:
    def __init__(self, model):
        self.model = model
        self.strategy_params = {
            "SMA_short": 20,
            "SMA_long": 50,
            "RSI_period": 14,
        }

    def calculate_indicators(self, data):
        data['SMA_short'] = data['close'].rolling(window=self.strategy_params["SMA_short"]).mean()
        data['SMA_long'] = data['close'].rolling(window=self.strategy_params["SMA_long"]).mean()
        data['RSI'] = self.calculate_rsi(data['close'], self.strategy_params["RSI_period"])
        return data

    def calculate_rsi(self, prices, n=14):
        deltas = np.diff(prices)
        seed = deltas[:n+1]
        up = seed[seed >= 0].sum() / n
        down = -seed[seed < 0].sum() / n
        rs = up / down
        rsi = np.zeros_like(prices)
        rsi[:n] = 100. - 100. / (1. + rs)

        for i in range(n, len(prices)):
            delta = deltas[i - 1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta

            up = (up * (n - 1) + upval) / n
            down = (down * (n - 1) + downval) / n

            rs = up / down
            rsi[i] = 100. - 100. / (1. + rs)

        return rsi

    def get_trade_signal(self, data):
        if data['SMA_short'].iloc[-1] > data['SMA_long'].iloc[-1] and data['RSI'].iloc[-1] < 70:
            return 'buy'
        elif data['SMA_short'].iloc[-1] < data['SMA_long'].iloc[-1] and data['RSI'].iloc[-1] > 30:
            return 'sell'
        else:
            return 'hold'

    def execute_trade(self, action, current_price, balance, positions):
        if action == 'buy':
            positions.append(current_price)
            balance -= current_price
        elif action == 'sell' and positions:
            bought_price = positions.pop(0)
            balance += current_price
        return balance, positions

    def update_strategy_params(self, new_params):
        self.strategy_params.update(new_params)
