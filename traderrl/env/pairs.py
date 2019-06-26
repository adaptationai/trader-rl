from .state_over_time import state_over_time_day
from ..common import Indicators
class PairDefault():
    def __init__(self, pair="AUD_USD", start=0, end=15, steps=288, spread=0.0002, half_spread=0.0001, load=True, live=False):
        self.live = live
        self.load = load
        self.pair = pair
        self.start = start
        self.end = end
        self.steps = steps
        self.spread = spread
        self.half_spread = half_spread
        self.random = False
        self.sequence = False
        self.indicators = Indicators()
    
    def state_over_time(self, state):
        s = state
        new_state = []
        stateold = state[-15:-1]
        stateold2 = state[-16:-2]
        so = self.indicators.stocastic_oscillator_fixed(s)
        sop = self.indicators.stocastic_oscillator_fixed(stateold)
        sop2 = self.indicators.stocastic_oscillator_fixed(stateold2)
        sma = self.indicators.sma(so,sop,sop2)
        #new_state.append([so, state[-1][0],state[-1][1], state[-1][2], state[-1][3], state[-1][4], state[-1][5], state[-1][6]])
        #new_state.append([so, state[-1][0],state[-1][1], state[-1][2], state[-1][3], state[-1][4], state[-1][5], state[-1][6], state[-2][0],state[-2][1], state[-2][2], state[-2][3], state[-2][4], state[-2][5], state[-2][6], state[-3][0],state[-3][1], state[-3][2], state[-3][3], state[-3][4], state[-3][5], state[-3][6], state[-4][0],state[-4][1], state[-4][2], state[-4][3], state[-4][4], state[-4][5], state[-4][6], state[-5][0],state[-5][1], state[-5][2], state[-5][3], state[-5][4], state[-5][5], state[-5][6], state[-6][0],state[-6][1], state[-6][2], state[-6][3], state[-6][4], state[-6][5], state[-6][6], state[-7][0],state[-7][1], state[-7][2], state[-7][3], state[-7][4], state[-7][5], state[-7][6], state[-8][0],state[-8][1], state[-8][2], state[-8][3], state[-8][4], state[-8][5], state[-8][6], state[-9][0],state[-9][1], state[-9][2], state[-9][3], state[-9][4], state[-9][5], state[-9][6] ,state[-10][0],state[-10][1], state[-10][2], state[-10][3], state[-10][4], state[-10][5], state[-10][6] ,state[-11][0],state[-11][1], state[-11][2], state[-11][3], state[-11][4], state[-11][5], state[-11][6], state[-12][0], state[-12][1], state[-12][2], state[-12][3], state[-12][4], state[-12][5], state[-12][6], state[-13][0], state[-13][1], state[-13][2], state[-13][3], state[-13][4], state[-13][5], state[-13][6], state[-14][0], state[-14][1], state[-14][2], state[-14][3], state[-14][4], state[-14][5], state[-14][6]])
        #for i in range(len(s)):
            #new_state.append(s[i][0])
            #new_state.append(s[i][1])
            #new_state.append(s[i][2])
            #new_state.append(s[i][3])
            #new_state.append(s[i][4])
            #new_state.append(s[i][5])
            #new_state.append(s[i][6])
        new_state.append([so, sop, sop2, sma])
        return new_state

    False
    



class PairEURUSD():
    def __init__(self):
        self.live = False
        self.load = False
        self.pair = "EUR_USD"
        self.start = 0
        self.end = 14
        self.steps = 30


    
    
