import os
import pickle
import joblib
import pandas as pd
import yaml

from fastapi import FastAPI
from pydantic import BaseModel


class CarData(BaseModel):
    odo: int
    khv: int
    vdk: int
    blg: int
    year: int


init_path = os.path.abspath(__file__)

def get_project_root(start_path):
    current = start_path
    while current != os.path.dirname(current):
        if os.path.exists(os.path.join(current, "requirements.txt")):
            return current
        current = os.path.dirname(current)
    raise FileNotFoundError("Project root was not found")

PROJECT_ROOT_PATH = get_project_root(init_path)

config_filepath = os.path.join(PROJECT_ROOT_PATH, 'configs/pipeline_config.yaml')

with open(config_filepath, 'r', encoding='utf-8') as f:
    config_data = yaml.safe_load(f)

OUTPUT_DIR_FOR_DATA = config_data['output_dir_for_data']
Y_SCALER_FILEPATH = os.path.join(OUTPUT_DIR_FOR_DATA, 'y_scaler.pkl')
X_SCALER_FILEPATH = os.path.join(OUTPUT_DIR_FOR_DATA, 'x_scaler.pkl')
MODEL_FILEPATH = os.path.join(PROJECT_ROOT_PATH, 'models/lr_ridge.pkl')

with open(X_SCALER_FILEPATH, 'rb') as f:
    x_scaler = pickle.load(f)

with open(Y_SCALER_FILEPATH, 'rb') as f:
    y_scaler = pickle.load(f)


app = FastAPI()
model = joblib.load(MODEL_FILEPATH)


@app.post('/predict_price')
def predict_price(data: CarData):
    features = [[data.odo, data.khv, data.vdk, data.blg, data.year]]
    columns = ["odo", "khv", "vdk", "blg", "year"]
    df = pd.DataFrame(data=features, columns=columns)

    transformed_features = x_scaler.transform(df.loc[:, :].values)
    transformed_df = pd.DataFrame(data=transformed_features, columns=columns)
    transformed_df['odo_sqr'] = transformed_df['odo'] ** 2

    configured_df = transformed_df.loc[:, ['odo', 'odo_sqr', 'khv', 'vdk', 'blg']].values
    odo_transformed = model.predict(configured_df)
    odo = int(y_scaler.inverse_transform(odo_transformed.reshape(-1, 1))[0][0])
    return {"odo": odo}