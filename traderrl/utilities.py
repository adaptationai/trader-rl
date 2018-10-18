import numpy as np
import random
from auth import Auth
import oandapyV20
import oandapyV20.endpoints.instruments as instruments


class DataGrabber():
    """gets data and processes ready to use"""

    def __init__(self):
        
        self.love = 14
        self.auth = Auth()
        self.client = oandapyV20.API(access_token=self.auth.access_token)

    def get_candles(self, _from,  count, granularity, instrument):
        params = {"from": _from, "count": count, "granularity": granularity}
        r = instruments.InstrumentsCandles(instrument=instrument, params=params)
        data = self.client.request(r)
        return data

    def data_converted(self, data):
        data_converted  = []
        for i in data['candles']:
            data_converted.append([i['mid']['c'], i['mid']['h'], i['mid']['l'], i['mid']['o']])
        return data_converted




