import numpy as np
import random
from .auth import Auth
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
        self.years = ['2017', '2016', '2015', '2014', '2013', '2012', '2011', '2010', '2009', '2008', '2007']
        self.instrument = ['EUR_USD', 'AUD_USD', 'GBP_USD', 'NZD_USD', 'USD_CHF', 'USD_CAD']
        self.time = ['00:00:00']
        self.hour = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23']
        self.minute = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32','33','34', '35', '36', '37', '38','39','40','41','42','43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59']
        self.day = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28']
        
        self.month = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
        self.granularity= ['M1', 'M5', 'M15', 'M30', 'H1', 'H4']
        #self.full_year = np.load('2016-1m.npy')
        #self.full_year = np.load('1year-5m.npy')
        #self.full_year = np.load('10year-1m.npy')
        #self.full_year = np.load('2018eval.npy') # eval
        #fulltimetest105.npy
        #self.full_year = np.load('eur10year192.npy') # full 10 year 194
        #self.full_year = np.load('fulltimetest105.npy')
        #self.full_year = np.load('all10year192.npy')
        #self.full_year = np.load('20year192.npy')
        self.full_year = np.load('evaleuro2018192.npy')
        
        
        '10year-194-15m.npy'

    def get_candles(self, _from,  count, granularity, instrument):
        params = {"from": _from, "count": count, "granularity": granularity}
        r = instruments.InstrumentsCandles(instrument=instrument, params=params)
        data = self.client.request(r)
        #print(data)
        return data

    def data_converted(self, data):
        data_converted  = []
        for i in data['candles']:
            #data_converted.append([i['mid']['c'], i['mid']['h'], i['mid']['l'], i['mid']['o']]) 
            data_converted.append([i['mid']['c'], i['mid']['h'], i['mid']['l'], i['mid']['o'], i['volume'], i['time']])

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
        year = random.choice(self.years)
        data = self.get_candles(year+'-01-01T00:00:00Z', 2880, "M1", "EUR_USD")
        data = self.data_converted(data)
        data = self.toarray(data)
        data = self.normalize(data)
        return data

    def process_to_array(self):
        year = random.choice(self.years)
        day = random.choice(self.day)
        month = random.choice(self.month)
        hour = random.choice(self.hour)
        minute = random.choice(self.minute)
        self.day = ['04']
        day = random.choice(self.day)
        #data = self.get_candles(year+'-'+month+'-'+day+'T'+hour+':'+minute+':00Z', 2880, "M1", "EUR_USD")
        #data = self.get_candles('2016-01-'+day+'T00:00:00Z', 2880, "M1", "EUR_USD")
        #data = self.get_candles('2016-'+month+'-'+day+'T00:00:00Z', 2880, "M1", "EUR_USD")
        data = self.get_candles('2017-02-09T00:00:00Z', 2880, "M1", "EUR_USD")
        data = self.data_converted(data)
        data = self.toarray(data)
        #np.save('state4.npy', data)
        data = self.difference(data)
        #full_data = []
        #full_data.append(data)
        #np.save('state60.npy', data)
        
        #print(difference)
        return data

    def process_to_array_2(self):
        #year = random.choice(self.years)
        #day = random.choice(self.day)
        #month = random.choice(self.month)
        #hour = random.choice(self.hour)
        #minute = random.choice(self.minute)
        #self.day = ['04']
        #day = random.choice(self.day)
        #data = self.get_candles(year+'-'+month+'-'+day+'T'+hour+':'+minute+':00Z', 2880, "M1", "EUR_USD")
        #data = self.get_candles('2016-01-'+day+'T00:00:00Z', 2880, "M1", "EUR_USD")
        full_data = []
        self.years = ['2018']
        #self.month = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11']
        self.month = ['01']
        self.instrument = ['EUR_USD']
        for i in self.instrument:
            for y in self.years:
                for m in self.month:
                    #if self.month == "02":
                        #day = self.day2
                    #else:
                        #day = self.day
                

                    for d in self.day:
                    
            

                        data = self.get_candles(y+'-'+m+'-'+d+'T00:00:00Z', 192, "M15", i)
                        #data = self.get_candles('2016-03-01T00:00:00Z', 2880, "M1", "EUR_USD")

                        data = self.data_converted(data)
                        data = self.time_to_array(data)
                    
                        #if data[0][5] != 5 or data[0][5] != 6:

                            #data[5] = data_time
                        data = self.toarray(data)
            #np.save('state4.npy', data)
                        #data = self.difference(data)
                    #full_data = []
                        full_data.append(data)
        np.save('evaleuro2018jan192.npy', full_data)
        
        #print(difference)
        return full_data

    def process_to_tensor(self):
        data = self.get_candles('2016-06-01T00:00:00Z', 2880, "M1", "EUR_USD")
        data = self.data_converted(data)
        data = self.toarray(dataall)
        data = self.normalize(daallta)
        data = self.totensor(datalla)
        return data

    def flatten_full(self, markeallt, user):
        market = data

        x = list()
        for i in range(len(old_data)):
            con = np.concatenate((data), axis=None)
            con = np.concatenate((con, old_data[i][1]), axis=None)
            con[0].tolist()
            x.append(con)
        return x

    def flatten(self, u, m, c):
        u = np.concatenate((u), axis=None)
        m = np.concatenate((m), axis=None)
        c = np.concatenate((c), axis=None)
        flattened = np.concatenate((m, u, c), axis=None)

        #k = self.data_grabber.flatten(market_details, player_details)
        return flattened

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
    def difference(self, state):
        new_state = []
        r = 194
        for i in range(96):
            before = state[i][0]
            b = i+1
            after = state[b][0]
            diff = after - before
            vol = state[b][4]
            o = state[b][3]
            l = state[b][2]
            h = state[b][1]

            new_state.append([after, diff, vol, o, l, h ])
        return new_state
    
    def difference2(self, state):
        #data_converted.append([i['mid']['c'], i['mid']['h'], i['mid']['l'], i['mid']['o'], i['volume']])
        new_state = []
        r = 194
        for i in range(194):
            c = state[i][0]
            h = state[i][1]
            l = state[i][2]
            o = state[i][3]
            v = state[i][4]
            c = c - o
            h = h - o
            l = l - o
            

            new_state.append([c, h, l, v])
        return new_state

    def load_state(self):
        daz = ['state01.npy', 'state02.npy', 'state03.npy', 'state04.npy', 'state05.npy', 'state06.npy', 'state07.npy', 'state08.npy', 'state09.npy', 'state10.npy', 'state11.npy', 'state12.npy', 'state13.npy', 'state14.npy', 'state15.npy', 'state16.npy', 'state17.npy', 'state18.npy', 'state19.npy', 'state20.npy', 'state21.npy', 'state22.npy', 'state23.npy', 'state24.npy', 'state25.npy', 'state26.npy', 'state27.npy', 'state28.npy', 'state29.npy', 'state30.npy']
        daz40 = ['state01.npy', 'state02.npy', 'state03.npy', 'state04.npy', 'state05.npy', 'state06.npy', 'state07.npy', 'state08.npy', 'state09.npy', 'state10.npy', 'state11.npy', 'state12.npy', 'state13.npy', 'state14.npy', 'state15.npy', 'state16.npy', 'state17.npy', 'state18.npy', 'state19.npy', 'state20.npy', 'state21.npy', 'state22.npy', 'state23.npy', 'state24.npy', 'state25.npy', 'state26.npy', 'state27.npy', 'state28.npy', 'state29.npy', 'state30.npy', 'state31.npy', 'state32.npy', 'state33.npy', 'state34.npy', 'state35.npy', 'state36.npy', 'state37.npy', 'state38.npy', 'state39.npy', 'state40.npy']
        daz60 = ['state01.npy', 'state02.npy', 'state03.npy', 'state04.npy', 'state05.npy', 'state06.npy', 'state07.npy', 'state08.npy', 'state09.npy', 'state10.npy', 'state11.npy', 'state12.npy', 'state13.npy', 'state14.npy', 'state15.npy', 'state16.npy', 'state17.npy', 'state18.npy', 'state19.npy', 'state20.npy', 'state21.npy', 'state22.npy', 'state23.npy', 'state24.npy', 'state25.npy', 'state26.npy', 'state27.npy', 'state28.npy', 'state29.npy', 'state30.npy', 'state31.npy', 'state32.npy', 'state33.npy', 'state34.npy', 'state35.npy', 'state36.npy', 'state37.npy', 'state38.npy', 'state39.npy', 'state40.npy', 'state41.npy', 'state42.npy', 'state43.npy', 'state44.npy', 'state45.npy', 'state46.npy', 'state47.npy', 'state48.npy', 'state49.npy', 'state50.npy', 'state51.npy', 'state52.npy', 'state53.npy', 'state54.npy', 'state55.npy', 'state56.npy', 'state57.npy', 'state58.npy', 'state59.npy', 'state60.npy']
        daz4 = ['state01.npy', 'state02.npy', 'state03.npy', 'state04.npy', 'state05.npy', 'state06.npy', 'state07.npy', 'state08.npy', 'state09.npy', 'state10.npy', 'state11.npy', 'state12.npy', 'state13.npy', 'state14.npy', 'state15.npy', 'state16.npy', 'state17.npy', 'state18.npy', 'state19.npy', 'state20.npy']
        daz72 = ['state01.npy']
        day = random.choice(daz60)
        print(day)
        data = np.load(day)
        return data

    def load_state_2(self):
        #daz = ['state01.npy', 'state02.npy', 'state03.npy', 'state04.npy', 'state05.npy', 'state06.npy', 'state07.npy', 'state08.npy', 'state09.npy', 'state10.npy', 'state11.npy', 'state12.npy', 'state13.npy', 'state14.npy', 'state15.npy', 'state16.npy', 'state17.npy', 'state18.npy', 'state19.npy', 'state20.npy', 'state21.npy', 'state22.npy', 'state23.npy', 'state24.npy', 'state25.npy', 'state26.npy', 'state27.npy', 'state28.npy', 'state29.npy', 'state30.npy']
        #daz40 = ['state01.npy', 'state02.npy', 'state03.npy', 'state04.npy', 'state05.npy', 'state06.npy', 'state07.npy', 'state08.npy', 'state09.npy', 'state10.npy', 'state11.npy', 'state12.npy', 'state13.npy', 'state14.npy', 'state15.npy', 'state16.npy', 'state17.npy', 'state18.npy', 'state19.npy', 'state20.npy', 'state21.npy', 'state22.npy', 'state23.npy', 'state24.npy', 'state25.npy', 'state26.npy', 'state27.npy', 'state28.npy', 'state29.npy', 'state30.npy', 'state31.npy', 'state32.npy', 'state33.npy', 'state34.npy', 'state35.npy', 'state36.npy', 'state37.npy', 'state38.npy', 'state39.npy', 'state40.npy']
        #daz60 = ['state01.npy', 'state02.npy', 'state03.npy', 'state04.npy', 'state05.npy', 'state06.npy', 'state07.npy', 'state08.npy', 'state09.npy', 'state10.npy', 'state11.npy', 'state12.npy', 'state13.npy', 'state14.npy', 'state15.npy', 'state16.npy', 'state17.npy', 'state18.npy', 'state19.npy', 'state20.npy', 'state21.npy', 'state22.npy', 'state23.npy', 'state24.npy', 'state25.npy', 'state26.npy', 'state27.npy', 'state28.npy', 'state29.npy', 'state30.npy', 'state31.npy', 'state32.npy', 'state33.npy', 'state34.npy', 'state35.npy', 'state36.npy', 'state37.npy', 'state38.npy', 'state39.npy', 'state40.npy', 'state41.npy', 'state42.npy', 'state43.npy', 'state44.npy', 'state45.npy', 'state46.npy', 'state47.npy', 'state48.npy', 'state49.npy', 'state50.npy', 'state51.npy', 'state52.npy', 'state53.npy', 'state54.npy', 'state55.npy', 'state56.npy', 'state57.npy', 'state58.npy', 'state59.npy', 'state60.npy']
        #daz4 = ['state01.npy', 'state02.npy', 'state03.npy', 'state04.npy', 'state05.npy', 'state06.npy', 'state07.npy', 'state08.npy', 'state09.npy', 'state10.npy', 'state11.npy', 'state12.npy', 'state13.npy', 'state14.npy', 'state15.npy', 'state16.npy', 'state17.npy', 'state18.npy', 'state19.npy', 'state20.npy']
        
        #day = random.choice()
        #print(day)
        #data = np.load('2016-1m.npy')
        day = random.choice(self.full_year)
        return day
    
    def time_to_array(self, data):
        for i in range(len(data)):
            date = data[i][5]
            
        #s = "2018-12-10T19:55:00.00000000Z"
            date = re.split('-|T|:|Z|', date)
            date = date[0:5]
        #print(s)
            date = list(map(int, date))
        
        #datetime.datetime.today()
        #datetime.datetime(s[0], s[1], s[2], s[3], s[4], 1, 173504))
            day = datetime.date(date[0], date[1], date[2]).weekday()
            hour = date[3]
            minute = date[4]
            date = [day, hour, minute]
            data[i][5] = day
            data[i].append(hour)
            data[i].append(minute)
        return data

    

    
#dates = ["2016", "2017", "2018"]
#test = DataGrabber()
#test.process_to_array_2()
#data = test.load_state_2()
#print(len(test.full_year[0]))
#print(len(data[1]))
#candles = test.get_candles('1998-06-01T00:00:00Z', 1, "M15", "EUR_USD")
#print(candles)
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
#state = [1,2,3,4,5,6,7,8,9,10]
#statenew = state[-4:]
#print(statenew )

