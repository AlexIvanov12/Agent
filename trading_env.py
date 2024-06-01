import numpy as np

class TradingEnv:
    def __init__(self, data):
        self.data = data
        self.current_step = 0
        self.balance = 100000  # Початковий баланс
        self.positions = []
        self.done = False

    def reset(self):
        self.current_step = 0
        self.balance = 100000
        self.positions = []
        self.done = False
        return self._next_observation()

    def _next_observation(self):
        # Отримання поточного стану ринку
        obs = np.array([
            self.data['close'].values[self.current_step],
            self.data['SMA_short'].values[self.current_step],
            self.data['SMA_long'].values[self.current_step],
            self.data['RSI'].values[self.current_step],
        ])
        return obs

    def step(self, action):
        # Виконання дії
        current_price = self.data['close'].values[self.current_step]
        if action == 0:  # Купити
            self.positions.append(current_price)
        elif action == 1 and self.positions:  # Продати
            bought_price = self.positions.pop(0)
            self.balance += current_price - bought_price

        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            self.done = True

        reward = self.balance
        obs = self._next_observation()
        return obs, reward, self.done, {}

    def render(self):
        # Візуалізація стану
        print(f'Step: {self.current_step}, Balance: {self.balance}, Positions: {self.positions}')
