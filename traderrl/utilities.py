import numpy as np
import random
from auth import Auth
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
from sklearn import preprocessing
import csv
import torch
import time
import cv2
import mss
import numpy


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
            #data_converted.append([i['volume'], i['time'],i['mid']['c'], i['mid']['h'], i['mid']['l'], i['mid']['o']])
        return data_converted


    def normalize(self, x):
        normalized = preprocessing.normalize(x)
        return normalized


    def scaled(self, x):
        scaled = preprocessing.scale(x)
        return scaled

    
    def tocsv(self, x, path):
        with open(path, "a", newline='') as fp:
            wr = csv.writer(fp, dialect='excel')
            for i in range(len(x)):
                wr.writerow(x[i])

    def totensor(self, data):
        data = torch.from_numpy(data)
        return data

    def batcher(self, x, y, batch_size):
        x_data = list()
        return x_data

    def toarray(self, x):
        x = np.array(x, dtype=np.float32)
        return x

    def process_to_normalized(self):
        data = self.get_candles('2016-01-01T00:00:00Z', 2880, "M1", "EUR_USD")
        data = self.data_converted(data)
        data = self.toarray(data)
        data = self.normalize(data)
        return data

    def process_to_array(self):
        data = self.get_candles('2016-01-01T00:00:00Z', 2880, "M1", "EUR_USD")
        data = self.data_converted(data)
        data = self.toarray(data)
        return data

    def process_to_tensor(self):
        data = self.get_candles('2016-01-01T00:00:00Z', 2880, "M1", "EUR_USD")
        data = self.data_converted(data)
        data = self.toarray(data)
        data = self.normalize(data)
        data = self.totensor(data)
        return data

    def get_screen(self):
        with mss.mss() as sct:
            # Part of the screen to capture
            monitor = {"top": 40, "left": 0, "width": 800, "height": 640}

            while "Screen capturing":
                last_time = time.time()

                # Get raw pixels from the screen, save it to a Numpy array
                img = numpy.array(sct.grab(monitor))

                # Display the picture
                #cv2.imshow("OpenCV/Numpy normal", img)

                # Display the picture in grayscale
                # cv2.imshow('OpenCV/Numpy grayscale',
                #            cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY))

                print("fps: {}".format(1 / (time.time() - last_time)))

                # Press "q" to quit
                if cv2.waitKey(25) & 0xFF == ord("q"):
                    cv2.destroyAllWindows()
                    break


    
#dates = ["2016", "2017", "2018"]
#test = DataGrabber()
#candles = test.get_candles(dates[0]+'-01-01T00:00:00Z', 2, "M1", "EUR_USD")
#some_data = test.data_converted(candles)
#some_data = test.toarray(some_data)
#some_data = test.normalize(some_data)
#some_data = test.totensor(some_data)
#data_day = some_data[0:1440]
#print(len(data_day))
#print(len(some_data))
#print(candles)
#print(some_data)
#test.get_screen()



