import pandas as pd
from ib_insync import *
from data_preparation import DataPreparation

class IBDataLoader:
    def __init__(self, host='127.0.0.1', port=7497, client_id=1):
        self.ib = IB()
        self.host = host
        self.port = port
        self.client_id = client_id

    def connect(self):
        self.ib.connect(self.host, self.port, self.client_id)

    def disconnect(self):
        self.ib.disconnect()

    def fetch_historical_data(self, symbol, exchange, end_date, duration, bar_size):
        contract = Stock(symbol, exchange)
        bars = self.ib.reqHistoricalData(
            contract,
            endDateTime=end_date,
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow='MIDPOINT',
            useRTH=True,
            formatDate=1)
        df = util.df(bars)
        return df

    def prepare_data(self, data):
        data_preparation = DataPreparation(data)
        prepared_data = data_preparation.prepare_data()
        return prepared_data

    def get_prepared_data(self, symbol, exchange, end_date, duration='1 D', bar_size='15 mins'):
        self.connect()
        raw_data = self.fetch_historical_data(symbol, exchange, end_date, duration, bar_size)
        self.disconnect()
        prepared_data = self.prepare_data(raw_data)
        return prepared_data



        
        