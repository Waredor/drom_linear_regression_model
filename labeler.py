import json
import pandas as pd

df = pd.DataFrame()

with open('prius_drom_khv_28_04_25.json', 'r', encoding='utf-8') as file:
    data_khv = json.load(file)

i = 0
for el in data_khv:
    for val in el.values():
        if val[2]['year'] is not None:
            df.loc[i, 'price'] = int(val[0]['price'])
            df.loc[i, 'odo'] = int(val[1]['odo'])
            df.loc[i, 'year'] = val[2]['year']
            df.loc[i, 'khv'] = 1
            df.loc[i, 'vdk'] = 0
            df.loc[i, 'blg'] = 0
    i += 1

with open('prius_drom_vdk_28_04_25.json', 'r', encoding='utf-8') as file:
    data_vdk = json.load(file)

i = 0
for el in data_vdk:
    for val in el.values():
        if val[2]['year'] is not None:
            df.loc[i, 'price'] = int(val[0]['price'])
            df.loc[i, 'odo'] = int(val[1]['odo'])
            df.loc[i, 'year'] = val[2]['year']
            df.loc[i, 'khv'] = 0
            df.loc[i, 'vdk'] = 1
            df.loc[i, 'blg'] = 0
    i += 1

with open('prius_drom_blg_28_04_25.json', 'r', encoding='utf-8') as file:
    data_blg = json.load(file)

i = 0
for el in data_blg:
    for val in el.values():
        if val[2]['year'] is not None:
            df.loc[i, 'price'] = int(val[0]['price'])
            df.loc[i, 'odo'] = int(val[1]['odo'])
            df.loc[i, 'year'] = val[2]['year']
            df.loc[i, 'khv'] = 0
            df.loc[i, 'vdk'] = 0
            df.loc[i, 'blg'] = 1
    i += 1

df = df[df['odo'] > 80000]
df['age'] = 2025 - df['year']
#df['odo_per_year'] = df['odo'] / (2025 - df['year'])
df = df.drop(columns=['year'])
df = df[df['price'] <= df['price'].quantile(0.99)]  # Удаляем 1% самых больших значений

print(f"Средняя цена: {df['price'].mean() / 10 ** 6} млн рублей")
print(f"Медианная цена: {df['price'].median() / 10 ** 6} млн рублей")

print(df)

df.to_csv('annotated_data_28_04_25.csv', index=False)