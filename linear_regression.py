import  pandas as pd
import matplotlib.pylab as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

df = pd.read_csv('annotated_data_CONC.csv')
X = df.iloc[:, 1:]
odo = df.iloc[:,1:2]
y = df.iloc[:, 0] / 10 ** 6

corr_matrix = df.corr()
print(corr_matrix)

X_train_mul, X_test_mul, y_train_mul, y_test_mul = train_test_split(X, y, test_size=0.15, random_state=42)
odo_train, odo_test, y_train, y_test = train_test_split(odo, y, test_size=0.15, random_state=42)

#print(df)

#simple lr non std
simple_lr = LinearRegression()

simple_lr.fit(odo_train, y_train)
predicted_y = simple_lr.predict(odo_test)
mse_simple_lr = mean_squared_error(y_test, predicted_y)

print(f'simple lr mse: {mse_simple_lr}')

plt.figure(figsize=(10, 6))
plt.scatter(odo, y)
plt.plot(odo_test, predicted_y, c = 'r')
plt.title('Scatter plot and a linear Regression Model')
plt.ylabel("price")
plt.xlabel("odo")
plt.show()

#multiple lr non std
mp_lr = LinearRegression()
mp_lr.fit(X_train_mul, y_train_mul)
predicted_y_mul = mp_lr.predict(X_test_mul)
mse_mul_lr = mean_squared_error(y_test_mul, predicted_y_mul)

print(f'multiple lr mse: {mse_mul_lr}')

#Сохраняем модели в файл

joblib.dump(mp_lr, 'ML_models/linear_regression_multiple.pkl')
joblib.dump(simple_lr, 'ML_models/linear_regression_odo.pkl')


