import numpy as np
#from utilities import DataGrabber
import random
import json
from oandapyV20 import API
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.positions as position
from oandapyV20.contrib.requests import MarketOrderRequest, LimitOrderRequest, MITOrderRequest, PositionCloseRequest
import time
from ..common import DataGrabber

from ..common import Auth


class MarketLive():
    def __init__(self):
        self.auth = Auth()
        self.accountID = self.auth.accountID 
        self.access_token = self.auth.access_token
        self.data_grabber = DataGrabber()

    def market_order_long(self):
        client = API(access_token=self.access_token)
        mo = MarketOrderRequest(instrument="EUR_USD", units=10000)
        r = orders.OrderCreate(self.accountID, data=mo.data)
        rv = client.request(r)

    def market_order_short(self):
        client = API(access_token=self.access_token)
        mo = MarketOrderRequest(instrument="EUR_USD", units=10000)
        r = orders.OrderCreate(self.accountID, data=mo.data)
        rv = client.request(r)

    def limit_order(self):
        client = API(access_token=self.access_token)
        ordr = LimitOrderRequest(instrument="EUR_USD", units=10000, price=1.08)
        r = orders.orderCreate(self.accountID, data=ordr.data)
        rv = client.request(r)

    def market_if_touched(self):
        client = API(access_token=self.access_token)
        ordr = MITOrderRequest(instrument="EUR_USD", units=10000, price=1.08)
        r = orders.OrderCreate(self.accountID, data=ordr.data)
        rv = client.request(r)

    def position_close_long(self):
        client = API(access_token=self.access_token)
        ordr = PositionCloseRequest(longUnits=10000)
        r = position.PositionClose(self.accountID, instrument="EUR_USD", data=ordr.data)
        rv = client.request(r)

    def position_close_short(self):
        client = API(access_token=self.access_token)
        ordr = PositionCloseRequest(longUnits=-10000)
        r = position.PositionClose(self.accountID, instrument="EUR_USD", data=ordr.data)
        rv = client.request(r)

    def candles_live(self):
        client = API(access_token=self.access_token)
        params = {"count": 1440, "granularity": "M1"}
        r = instruments.InstrumentsCandles(instrument="EUR_USD", params=params)
        data = client.request(r)
        data = self.data_grabber.data_converted(data)
        data = self.data_grabber.time_to_array(data)
        data = self.data_grabber.toarray(data)
        return data

    def live_step_delay(self):
        current_time = time.time()
        time_to_sleep = 320 - (current_time % 320)
        time.sleep(time_to_sleep)
