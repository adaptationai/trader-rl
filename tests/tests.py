import unittest
from .traderrl import DataGrabber


class DataTest(unittest.TestCase):
    def test(self):
        test = DataGrabber()
        candles = test.get_candles("2016-01-01T00:00:00Z", 1440, "M1", "EUR_USD")
        some_data = test.data_converted(candles)
        print(some_data)
        print(len(some_data))

test = DataTest()
test.test()
if __name__ == '__main__':
    unittest.main()



