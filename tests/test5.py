import time
print(time.time())
current_time = time.time()
time_to_sleep = 320 - (current_time % 320)
time.sleep(time_to_sleep)
print(time.time())
