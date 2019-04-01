import numpy as np
#from utilities import DataGrabber
import random
import json
from oandapyV20 import API
import oandapyV20.endpoints.forexlabs as labs
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.positions as position
from oandapyV20.contrib.requests import MarketOrderRequest, LimitOrderRequest, MITOrderRequest, PositionCloseRequest, StopLossDetails, TakeProfitDetails, TrailingStopLossDetails
import time
from .utilities import DataGrabber

from .auth import Auth


class MarketLive():
    def __init__(self):
        self.auth = Auth()
        self.accountID = self.auth.accountID 
        self.access_token = self.auth.access_token
        self.data_grabber = DataGrabber()

    def market_order_long(self, pair="EUR_USD", unit=10000):
        client = API(access_token=self.access_token)
        mo = MarketOrderRequest(instrument=pair, units=unit)
        r = orders.OrderCreate(self.accountID, data=mo.data)
        rv = client.request(r)

    def market_order_short(self, pair="EUR_USD", unit=-10000):
        client = API(access_token=self.access_token)
        mo = MarketOrderRequest(instrument=pair, units=unit)
        r = orders.OrderCreate(self.accountID, data=mo.data)
        rv = client.request(r)

    def limit_order(self, pair="EUR_USD", unit=10000, p=1.08):
        client = API(access_token=self.access_token)
        ordr = LimitOrderRequest(instrument=pair, units=10000, price=p)
        r = orders.orderCreate(self.accountID, data=ordr.data)
        rv = client.request(r)

    def market_if_touched(self, pair="EUR_USD", unit=10000, p=1.08):
        client = API(access_token=self.access_token)
        ordr = MITOrderRequest(instrument=pair, units=unit, price=p)
        r = orders.OrderCreate(self.accountID, data=ordr.data)
        rv = client.request(r)

    def position_close_long(self, pair="EUR_USD", unit=10000):
        client = API(access_token=self.access_token)
        ordr = PositionCloseRequest(longUnits=unit)
        r = position.PositionClose(self.accountID, instrument=pair, data=ordr.data)
        rv = client.request(r)

    def position_close_short(self, pair="EUR_USD", unit=10000):
        client = API(access_token=self.access_token)
        ordr = PositionCloseRequest(shortUnits=unit)
        r = position.PositionClose(self.accountID, instrument=pair, data=ordr.data)
        rv = client.request(r)

    def candles_live(self, pair="EUR_USD", count=1440, gran="M1"):
        client = API(access_token=self.access_token)
        params = {"count": count, "granularity": gran}
        r = instruments.InstrumentsCandles(instrument=pair, params=params)
        data = client.request(r)
        data = self.data_grabber.data_converted(data)
        data = self.data_grabber.time_to_array(data)
        data = self.data_grabber.toarray(data)
        return data

    def live_step_delay(self):
        current_time = time.time()
        time_to_sleep = 320 - (current_time % 320)
        #time_to_sleep = 60 - (current_time % 60)
        time.sleep(time_to_sleep)


    def market_order_long_sl_tp_ts(self, sl, tp, ts, pair="EUR_USD", unit=10000):
        client = API(access_token=self.access_token)
        stopLossOnFill = StopLossDetails(price=sl)
        takeProfitOnFillOrder = TakeProfitDetails(price=tp)
        trailingStopLossOnFill = TrailingStopLossDetails(price=ts)
        ordr = MarketOrderRequest(
            instrument=pair, 
            units=unit, 
            stopLossOnFill=stopLossOnFill.data, 
            takeProfitOnFill=takeProfitOnFillOrder.data, 
            trailingStopLossOnFill=trailingStopLossOnFill.data
        )
        r = orders.OrderCreate(self.accountID, data=ordr.data)
        rv = client.request(r)

    def market_order_short_sl_tp_ts(self, sl, tp, ts, pair="EUR_USD", unit=-10000):
        client = API(access_token=self.access_token)
        stopLossOnFill = StopLossDetails(price=sl)
        takeProfitOnFillOrder = TakeProfitDetails(price=tp)
        trailingStopLossOnFill = TrailingStopLossDetails(price=ts)
        ordr = MarketOrderRequest(
            instrument=pair, 
            units=unit, 
            stopLossOnFill=stopLossOnFill.data, 
            takeProfitOnFill=takeProfitOnFillOrder.data, 
            trailingStopLossOnFill=trailingStopLossOnFill.data
        )
        r = orders.OrderCreate(self.accountID, data=ordr.data)
        rv = client.request(r)

    def market_order_long_sl_tp(self, sl, tp, pair="EUR_USD", unit=10000):
        client = API(access_token=self.access_token)
        stopLossOnFill = StopLossDetails(price=sl)
        takeProfitOnFillOrder = TakeProfitDetails(price=tp)
        ordr = MarketOrderRequest(
            instrument=pair, 
            units=unit, 
            stopLossOnFill=stopLossOnFill.data, 
            takeProfitOnFill=takeProfitOnFillOrder.data
        )
        r = orders.OrderCreate(self.accountID, data=ordr.data)
        rv = client.request(r)

    def market_order_short_sl_tp(self, sl, tp, pair="EUR_USD", unit=-10000):
        client = API(access_token=self.access_token)
        stopLossOnFill = StopLossDetails(price=sl)
        takeProfitOnFillOrder = TakeProfitDetails(price=tp)
        ordr = MarketOrderRequest(
            instrument=pair, 
            units=unit, 
            stopLossOnFill=stopLossOnFill.data, 
            takeProfitOnFill=takeProfitOnFillOrder.data
        )
        r = orders.OrderCreate(self.accountID, data=ordr.data)
        rv = client.request(r)

    def market_spread(self, pair="EUR_USD"):
        #Average spread 15 minute interval
        client = API(access_token=self.access_token)
        params = {
          "instrument": pair,
          "period": 0
        }
        r = labs.Spreads(params=params) 
        spread = client.request(r)
        spread = spread['avg']
        spread = spread[0][1]
        print(spread)
        return float(spread)


