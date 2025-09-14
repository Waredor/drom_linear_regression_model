import os
import pickle
import json
import yaml
import pandas as pd
import matplotlib.pylab as plt

from utils.parser import Parser
from sklearn.preprocessing import MinMaxScaler


# FILEPATHS INIT
config_filepath = 'configs/pipeline_config.yaml'
with open(config_filepath, 'r', encoding='utf-8') as f:
    config_data = yaml.safe_load(f)

OUTPUT_DIR_FOR_DATA = config_data['output_dir_for_data']
ANNOTATED_DATA_FILEPATH = os.path.join(OUTPUT_DIR_FOR_DATA, 'annotated_data.csv')
Y_SCALER_FILEPATH = os.path.join(OUTPUT_DIR_FOR_DATA, 'y_scaler.pkl')
X_SCALER_FILEPATH = os.path.join(OUTPUT_DIR_FOR_DATA, 'x_scaler.pkl')
ODO_SCALER_FILEPATH = os.path.join(OUTPUT_DIR_FOR_DATA, 'odo_scaler.pkl')


# REQUESTS
HEADERS = config_data['headers']


# DATA
CLASS_NAME = config_data['class_name']
URL_DICT = config_data['url_dict']


# CONFIG PARSER
parser = Parser(
    req_headers=HEADERS,
    base_url_dict=URL_DICT,
    class_name=CLASS_NAME,
    output_filepath=OUTPUT_DIR_FOR_DATA
)


# COLLECT DATA
parser.collect_elements()


# TRANSFORM DATA
annotated_data = []

df = pd.DataFrame()

files = [el for el in os.listdir(OUTPUT_DIR_FOR_DATA) if el.endswith('.json')]

for file in files:
    filepath = os.path.join(OUTPUT_DIR_FOR_DATA, file)
    with open(filepath, 'r', encoding='utf-8') as f:
        if file.split('.')[0] == 'khv':
            data_khv = json.load(f)

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

        elif file.split('.')[0] == 'vdk':
            data_vdk = json.load(f)

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

        elif file.split('.')[0] == 'blg':
            data_blg = json.load(f)

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

        else:
            raise ValueError(f"filename {f} is not correct!")


# FEATURE ENGINEERING
x_data = df.loc[:, ['odo', 'khv', 'vdk', 'blg', 'year']]
y_data = df.loc[:, ['price']]

features_data = x_data.values
target_data = y_data.values
odo_data = x_data.loc[:, ['odo']].values

x_columns = x_data.columns
y_columns = y_data.columns

x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()
odo_scaler = MinMaxScaler()

x_data_transform = x_scaler.fit_transform(features_data)
y_data_transform = y_scaler.fit_transform(target_data)
odo_data_transform = odo_scaler.fit_transform(odo_data)

x_df = pd.DataFrame(data=x_data_transform, columns=x_columns)
y_df = pd.DataFrame(data=y_data_transform, columns=y_columns)

df = pd.concat([y_df, x_df], axis=1)

odo_quantile_95 = df['odo'].quantile(0.95)
odo_quantile_5 = df['odo'].quantile(0.05)
df = df[(df['odo'] > odo_quantile_5) & (df['odo'] < odo_quantile_95)]
df['odo_sqr'] = df['odo'] ** 2
df['odo_cub'] = df['odo'] ** 3


# SAVE ANNOTATED DATA
df.to_csv(ANNOTATED_DATA_FILEPATH, index=False)


# SAVE SCALER
with open(Y_SCALER_FILEPATH, 'wb') as f:
    pickle.dump(y_scaler, f)

with open(X_SCALER_FILEPATH, 'wb') as f:
    pickle.dump(x_scaler, f)

with open(ODO_SCALER_FILEPATH, 'wb') as f:
    pickle.dump(odo_scaler, f)


# CREATE SCATTER PLOT
odo = df['odo']
target = df['price']

plt.figure(figsize=(10, 6))
plt.scatter(odo, target)
plt.title('Scatter plot and a linear Regression Model')
plt.ylabel("price")
plt.xlabel("odo")
plt.show()