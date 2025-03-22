import json
import random
import pandas as pd

annotated_data = []

with open('prius_drom_khv.json', 'r', encoding='utf-8') as file:
    data_khv = json.load(file)

for el in data_khv:
    for val in el.values():
        price = val[0]['price']
        odo = int(val[1]['odo'])
        year = val[2]['year']
        if year is not None and odo > 80000:
            annotated_el = {'price': price, 'odo': odo, 'year': year, 'khv': 1, 'vdk': 0, 'blg': 0}
            annotated_data.append(annotated_el)

with open('prius_drom_vdk.json', 'r', encoding='utf-8') as file:
    data_vdk = json.load(file)

for el in data_vdk:
    for val in el.values():
        price = val[0]['price']
        odo = int(val[1]['odo'])
        year = val[2]['year']
        if year is not None and odo > 80000:
            annotated_el = {'price': price, 'odo': odo, 'year': year, 'khv': 0, 'vdk': 1, 'blg': 0}
            annotated_data.append(annotated_el)

with open('prius_drom_blg.json', 'r', encoding='utf-8') as file:
    data_blg = json.load(file)

for el in data_blg:
    for val in el.values():
        price = val[0]['price']
        odo = int(val[1]['odo'])
        year = val[2]['year']
        if year is not None and odo > 80000:
            annotated_el = {'price': price, 'odo': odo, 'year': year, 'khv': 0, 'vdk': 0, 'blg': 1}
            annotated_data.append(annotated_el)

random.shuffle(annotated_data)
with open('annotated_data.json', 'w') as file:
    json.dump(annotated_data, file, ensure_ascii=False)

df = pd.read_json('annotated_data.json')
mean = df['price'].mean()
std = df['price'].std()
print(mean, std)
threshold = 2
#df = df[(df['price'] < mean + threshold * std) & (df['price'] > mean - threshold * std)]
print(df)
df.to_csv('annotated_data.csv', index=False)