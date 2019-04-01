from .state_over_time import state_over_time_day
class PairDefault():
    def __init__(self, pair="EUR_USD", start=0, end=14, steps=30, load=True, live=False):
        self.live = live
        self.load = load
        self.pair = pair
        self.start = start
        self.end = end
        self.steps = steps
        self.random = False
        self.sequence = False
    
    def state_over_time(self, state):
        new_state = state_over_time_day(state)
        return state

    
    



class PairEURUSD():
    def __init__(self):
        self.live = False
        self.load = False
        self.pair = "EUR_USD"
        self.start = 0
        self.end = 14
        self.steps = 30


    
    
