import  pandas as pd
import matplotlib.pylab as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import joblib

df = pd.read_csv('annotated_data.csv')
X = df.loc[:,['odo', 'khv', 'vdk', 'blg']].values
odo = df.iloc[:,1:2].values
target = df.iloc[:,0:1].values

#print(df)
#simple lr non std
simple_lr = LinearRegression()

mse = cross_val_score(simple_lr,
                      odo,
                      target,
                      scoring='neg_mean_squared_error',
                      cv=5)

print(f'non std_simple lr: {mse.mean()}')

simple_lr.fit(odo, target)
predicted_y = simple_lr.predict(odo)

plt.figure(figsize=(10, 6))
plt.scatter(odo, target)
plt.plot(odo, predicted_y, c = 'r')
plt.title('Scatter plot and a linear Regression Model')
plt.ylabel("price")
plt.xlabel("odo")
plt.show()

#multiple lr non std
mp_lr = LinearRegression()
mp_lr.fit(X, target)

mse = cross_val_score(mp_lr,
                     X,
                     target,
                     scoring='neg_mean_squared_error',
                     cv=5)

print(f'non std_multiple lr: {mse.mean()}')

y_mp = mp_lr.predict([[200000,0,1,0]])
print(y_mp)


#Сохраняем модели в файл

#joblib.dump(mp_lr, 'linear_ML_models/linear_regression_multiple.pkl')
#joblib.dump(simple_lr, 'linear_ML_models/linear_regression_odo.pkl')


