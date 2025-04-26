import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBRegressor, DMatrix, train
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import joblib

# Проверка версии XGBoost
print(f"Версия XGBoost: {xgb.__version__}")

# Загрузка данных
df = pd.read_csv('annotated_data.csv')
X = df.iloc[:, 1:]
#X = X.drop(columns=['khv', 'blg', 'vdk'])
y = df.iloc[:, 0] / 10 ** 6

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Логарифмирование целевой переменной
y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)

# Определение базовой модели XGBoost для RandomizedSearchCV
base_model = XGBRegressor(random_state=42, n_jobs=-1)

# Определение пространства гиперпараметров для поиска
param_dist = {
    'n_estimators': [100, 200, 300, 400, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 10, 12],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

# Настройка RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_dist,
    n_iter=50,
    scoring='neg_mean_squared_error',
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

# Выполнение поиска гиперпараметров на логарифмированных данных
random_search.fit(X_train, y_train_log)

# Вывод лучших гиперпараметров
print("Лучшие гиперпараметры:", random_search.best_params_)
print("Лучший MSE на кросс-валидации (логарифмированный):", -random_search.best_score_)

# Подготовка данных для xgboost.train
dtrain = DMatrix(X_train, label=y_train_log)
dtest = DMatrix(X_test, label=y_test_log)

# Параметры для xgboost.train
params = {
    'subsample': random_search.best_params_['subsample'],
    'n_estimators': random_search.best_params_['n_estimators'],
    'min_child_weight': random_search.best_params_['min_child_weight'],
    'max_depth': random_search.best_params_['max_depth'],
    'learning_rate': random_search.best_params_['learning_rate'],
    'colsample_bytree': random_search.best_params_['colsample_bytree'],
    'random_state': 42,
    'n_jobs': -1
}

# Настройка ранней остановки через callbacks
early_stopping = xgb.callback.EarlyStopping(
    rounds=10,
    metric_name='rmse',
    data_name='eval',
    maximize=False
)

# Обучение модели с использованием xgboost.train
bst = xgb.train(
    params,
    dtrain,
    num_boost_round=params['n_estimators'],
    evals=[(dtest, 'eval')],
    callbacks=[early_stopping],
    verbose_eval=False
)

# Предсказание на тестовой выборке
y_pred_test_log = bst.predict(DMatrix(X_test))
y_pred_test = np.expm1(y_pred_test_log)
mse = mean_squared_error(y_test, y_pred_test)
print(f"MSE на тестовой выборке: {mse}")

# Вычисление RMSLE
rmsle = np.sqrt(mean_squared_error(y_test_log, y_pred_test_log))
print(f"RMSLE на тестовой выборке: {rmsle}")

# Важность признаков
feature_importance = pd.Series(bst.get_score(importance_type='weight'), index=X.columns).sort_values(ascending=False)
print("\nВажность признаков:")
print(feature_importance)

joblib.dump(bst, 'ML_models/XGBoost_model.pkl')