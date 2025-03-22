import matplotlib.pylab as plt
import pandas as pd
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv('annotated_data.csv')
X = df.iloc[:,1:]
odo = df.iloc[:,1:2]
target = df.iloc[:,0:1].values

# Создание базовой модели
base_model = DecisionTreeRegressor(max_depth=10)

# Обучение модели Bagging
bagging_model = BaggingRegressor(estimator=base_model, n_estimators=100, random_state=10)
bagging_model.fit(odo, target)

# Предсказания
y_pred_bagging = bagging_model.predict(odo)

plt.figure(figsize=(10, 6))
plt.scatter(odo, target)
plt.plot(odo, y_pred_bagging, c = 'r')
plt.title('Scatter plot and a Decision Tree Regression Model')
plt.ylabel("price")
plt.xlabel("odo")
plt.show()