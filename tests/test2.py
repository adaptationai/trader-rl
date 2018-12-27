import re

s = "2018-12-10T19:55:00.00000000Z"
s = re.split('-|T|:|Z|', s)
s = s[0:5]
print(s)
s = list(map(int, s))
import datetime
datetime.datetime.today()
print(datetime.datetime(s[0], s[1], s[2], s[3], s[4], 1, 173504))
print(datetime.date(s[0], s[1], s[2]).weekday())