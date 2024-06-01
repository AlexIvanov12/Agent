import pandas as pd
import numpy as np


class DataPreparation:
    def __init__(self, data):
        self.data = data


    def clean_data(self):
        #delate 
        self.data.dropna(inplace = True)
        return self.data
    

    def remove_anomalies(self):
         self.data = self.data[(self.data['close'] > self.data['close'].quantile(0.01)) & (self.data['close'] < self.data['close'].quantile(0.99))]
         return self.data
    

    def normalize_data(self):
        self.data['close'] = (self.data['close'] - self.data['close'].min()) / (self.data['close'].max() - self.data['close'].min())
        return self.data
    

    def calculate_indicators(self):
        self.data['SMA_20'] = self.data['close'].rolling(window = 20).mean()
        self.data['SMA_50'] = self.data['close'].rolling(window = 50).mean()
        self.data['RSI'] = self.calculate_rsi(self.data['close'])
        return self.data
    

    def calculate_rsi(self, prices, n =14):
        deltas = np.diff(prices)
        seed = deltas[:n+1]
        up = seed[seed >= 0].sum() / n
        down = seed[seed < 0].sum() / n
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

            up = (up * (n -1) + upval) / n
            down = (down * (n - 1) + downval) / n

            rs = up / down
            rsi[i] = 100. - 100. / (1. + rs)
        return rsi
    

    def create_lag_features(self, lags = 5):
        for lag in range(1, lags + 1):
            self.data[f'close_lag_{lag}'] = self.data['close'].shift(lag)
        self.data.dropna(inplace = True)
        return self.data
    

    def prepare_data(self):
        self.clean_data()
        self.remove_anomalies()
        self.normalize_data()
        self.calculate_indicators()
        self.create_lag_features()
        return self.data