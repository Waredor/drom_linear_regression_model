import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, DMatrix, train
import xgboost as xgb

# Загрузка данных
df = pd.read_csv('annotated_data_CONC.csv')

# Перемешивание строк датафрейма
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Создание целевой переменной city
df['city'] = np.select(
    [df['khv'] == 1, df['blg'] == 1, df['vdk'] == 1],
    [0, 1, 2],  # 0: khv, 1: blg, 2: vdk
)

# Удаление ненужных столбцов
df = df.drop(columns=['khv', 'blg', 'vdk'])

print(df)

# Формирование признаков и целевой переменной
X = df.iloc[:, 0:3]
y = df.iloc[:, 3]

# Добавление нелинейных признаков
X['age_squared'] = X['age'] ** 2
X['log_odo'] = np.log1p(X['odo'])
X['price_squared'] = X['price'] ** 2

# Масштабирование признаков
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)

# Вычисление весов классов
class_weights = 1 / df['city'].value_counts(normalize=True)
weights = df['city'].map(class_weights).values

# Разделение данных
X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
    X, y, weights, test_size=0.3, random_state=42, stratify=y
)

# Проверка распределения классов
print("Распределение классов в обучающей выборке:", pd.Series(y_train).value_counts())
print("Распределение классов в тестовой выборке:", pd.Series(y_test).value_counts())

# Создание базовой модели
base_model = XGBClassifier(
    objective='multi:softmax',
    num_class=3,
    random_state=42,
    n_jobs=-1
)

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
    scoring='f1_macro',
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

# Выполнение поиска гиперпараметров с весами
random_search.fit(X_train, y_train, sample_weight=weights_train)

# Вывод лучших гиперпараметров
print("Лучшие гиперпараметры:", random_search.best_params_)
print("Лучший F1-score на кросс-валидации:", random_search.best_score_)

# Подготовка данных для xgboost.train
dtrain = DMatrix(X_train, label=y_train, weight=weights_train)
dtest = DMatrix(X_test, label=y_test, weight=weights_test)

# Параметры для xgboost.train
params = {
    'objective': 'multi:softmax',
    'num_class': 3,
    'subsample': random_search.best_params_['subsample'],
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
    metric_name='mlogloss',
    data_name='eval',
    maximize=False
)

# Обучение модели с использованием xgboost.train
bst = xgb.train(
    params,
    dtrain,
    num_boost_round=random_search.best_params_['n_estimators'],
    evals=[(dtest, 'eval')],
    callbacks=[early_stopping],
    verbose_eval=False
)

# Предсказание на тестовой выборке
y_pred = bst.predict(DMatrix(X_test)).astype(int)

# Оценка модели
print("Отчёт классификации:")
print(classification_report(y_test, y_pred, target_names=['khv', 'blg', 'vdk']))

print("Матрица ошибок:")
print(confusion_matrix(y_test, y_pred))

accuracy = (y_pred == y_test).mean()
print(f"Точность (accuracy) на тестовой выборке: {accuracy:.4f}")

# Важность признаков
feature_importance = pd.Series(bst.get_score(importance_type='weight'), index=X.columns).sort_values(ascending=False)
print("\nВажность признаков:")
print(feature_importance)