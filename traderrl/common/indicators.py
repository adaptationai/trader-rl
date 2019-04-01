class Indicators():
    def __init__(self):
        self.ramona = 'Love'

    def fibonacci(self, candle, price):
        price_max = candle[1]
        price_min = candle[2]
        diff = price_max - price_min
        levels = [0,0,0]
        level1 = price_max - 0.236 * diff
        level2 = price_max - 0.382 * diff
        level3 = price_max - 0.681 * diff
        return diff

    
    def stocastic_oscillator(self, candle):
        c = candle[0]
        h = candle[1]
        l = candle[2]
        o = candle[3]
        p = (c - l) / (h - l)
        return p

    def stocastic_oscillator_fixed(self, candles):
        #This is a basic version. Will implement better version later
        #Actually it is shit hehe
        #%K = (Current Close - Lowest Low)/(Highest High - Lowest Low) * 100
        #%D = 3-day SMA of %K
        #Lowest Low = lowest low for the look-back period
        #Highest High = highest high for the look-back period
        candles = candles[-14:]
        close = []
        high = []
        low = []
        op = []
        for i in range(len(candles)):
            close.append(candles[i][0])
            high.append(candles[i][1])
            low.append(candles[i][2])
            op.append(candles[i][3])
        c = close[-1]
        h = np.max(high) 
        l = np.min(low)
        p = (c - l) / (h - l) * 100
        return p
