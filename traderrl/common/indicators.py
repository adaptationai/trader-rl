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

