import os
import pickle
import logging
import  pandas as pd
import yaml
import joblib
import matplotlib.pylab as plt
import numpy as np

from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split


# FILEPATHS INIT
config_filepath = 'configs/pipeline_config.yaml'
with open(config_filepath, 'r', encoding='utf-8') as f:
    config_data = yaml.safe_load(f)

OUTPUT_DIR_FOR_DATA = config_data['output_dir_for_data']
ANNOTATED_DATA_FILEPATH = os.path.join(OUTPUT_DIR_FOR_DATA, 'annotated_data.csv')
Y_SCALER_FILEPATH = os.path.join(OUTPUT_DIR_FOR_DATA, 'y_scaler.pkl')
ODO_SCALER_FILEPATH = os.path.join(OUTPUT_DIR_FOR_DATA, 'odo_scaler.pkl')

df = pd.read_csv(ANNOTATED_DATA_FILEPATH)
X = df.loc[:, ['odo', 'khv', 'vdk', 'blg', 'year']].values
odo = df.loc[:, ['odo']].values
odo_fit = df.loc[:, ['odo', 'odo_sqr', 'khv', 'vdk', 'blg']].values

target = df['price'].values


with open(Y_SCALER_FILEPATH, 'rb') as f:
    y_scaler = pickle.load(f)

with open(ODO_SCALER_FILEPATH, 'rb') as f:
    odo_scaler = pickle.load(f)


# LOGGER INIT
stream_handler = logging.StreamHandler()
logger = logging.getLogger('linear_regression_pipeline_logger')
logger.setLevel(logging.INFO)
logger.addHandler(stream_handler)


# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(odo_fit, target, train_size=0.8, random_state=42)
X_train_plot, X_test_plot, y_train_plot, y_test_plot = train_test_split(odo, target, train_size=0.8, random_state=42)

y_test = y_scaler.inverse_transform(y_test.reshape(-1, 1))
target_plot = y_scaler.inverse_transform(y_train.reshape(-1, 1))

odo_plot_scatter = odo_scaler.inverse_transform(X_train_plot)
odo_plot_model = odo_scaler.inverse_transform(X_test_plot)
odo_plot_model_for_sorting = X_test[:, 0]

sort_indices = np.argsort(odo_plot_model_for_sorting)
odo_sorted = odo_plot_model[sort_indices]


# RIDGE LR
lr_ridge = linear_model.Ridge(alpha=0.005)
lr_ridge.fit(X_train, y_train)
y_pred = lr_ridge.predict(X_test)

y_pred_ridge_lr = y_scaler.inverse_transform(y_pred.reshape(-1, 1))
y_pred_sorted_ridge_lr = y_pred_ridge_lr[sort_indices]

rmse = root_mean_squared_error(y_test, y_pred_ridge_lr)

logger.info(f'Ridge linear regression RMSE: {rmse}')


# LASSO LR
lr_lasso = linear_model.Lasso(alpha=0.0005)
lr_lasso.fit(X_train, y_train)
y_pred = lr_lasso.predict(X_test)

y_pred_lasso_lr = y_scaler.inverse_transform(y_pred.reshape(-1, 1))
y_pred_sorted_lasso_lr = y_pred_lasso_lr[sort_indices]

rmse = root_mean_squared_error(y_test, y_pred_lasso_lr)

logger.info(f'Lasso linear regression RMSE: {rmse}')


# DECISION TREE
dt = DecisionTreeRegressor(max_depth=7, min_samples_leaf=10)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

y_pred_dt_lr = y_scaler.inverse_transform(y_pred.reshape(-1, 1))
y_pred_sorted_dt_lr = y_pred_dt_lr[sort_indices]

rmse = root_mean_squared_error(y_test, y_pred_dt_lr)

logger.info(f'Decision Tree RMSE: {rmse}')


# SAVE MODELS
joblib.dump(lr_ridge, 'models/lr_ridge.pkl')
joblib.dump(lr_lasso, 'models/lr_lasso.pkl')
joblib.dump(dt, 'models/decision_tree.pkl')


# CREATE PLOTS
plt.figure(figsize=(10, 6))
plt.scatter(odo_plot_scatter, target_plot)
plt.plot(odo_sorted, y_pred_sorted_ridge_lr, c = 'r')
plt.title('Scatter plot and Ridge linear Regression Model')
plt.ylabel("price")
plt.xlabel("odo")
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(odo_plot_scatter, target_plot)
plt.plot(odo_sorted, y_pred_sorted_lasso_lr, c = 'r')
plt.title('Scatter plot and Lasso linear Regression Model')
plt.ylabel("price")
plt.xlabel("odo")
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(odo_plot_scatter, target_plot)
plt.plot(odo_sorted, y_pred_sorted_dt_lr, c = 'r')
plt.title('Scatter plot and Decision tree linear Regression Model')
plt.ylabel("price")
plt.xlabel("odo")
plt.show()