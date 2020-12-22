import keyboard as k
import json
import pandas as pd
import sys
import datetime
import csv
import time

if sys.argv[1] == 'c':
	is_cat=True
	print('Reading in Cat mode')
elif sys.argv[1] == 'h':
	is_cat = False
	print('Reading in Human mode')
else: 
	print('Unknow read mode!')

print('Reading keyboard input')
recorded = k.record(until='esc')
keyboard_events = []
keys_pressed = ''
scan_codes = []

for key in recorded:
	temp = json.loads(key.to_json())
	keys_pressed += temp['name'] +' '
	scan_codes.append(temp['scan_code'])
	keyboard_events.append(temp)

recorded_keys = {'input': keys_pressed, 'is_cat': is_cat}
recorded_scan_codes = {'input': scan_codes, 'is_cat': is_cat}
file_num = time.time()
if is_cat:
	file_path = f'data/cat/catInput-{file_num}.csv'
else:
	file_path = f'data/human/humanInput-{file_num}.csv'

with open(file_path, 'a') as f:
	writer = csv.writer(f,delimiter=',', lineterminator='\n')
	writer.writerow([keys_pressed])


# with open('data/codesInput.csv', 'a') as f:
# 	writer = csv.writer(f,delimiter=',', lineterminator='\n')
# 	writer.writerow([scan_codes, is_cat])

print(keys_pressed)
# 23xevents.to_csv('data/catInput.csv', mode='a', header=False)
