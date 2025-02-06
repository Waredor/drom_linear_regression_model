import  pandas as pd
import matplotlib.pylab as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import joblib

df = pd.read_csv('annotated_data.csv')
odo = df.iloc[:,1:2]
target = df.iloc[:,0:1]
simple_lr = LinearRegression()

simple_lr.fit(odo, target)

predicted_y = simple_lr.predict(odo)

plt.figure(figsize=(10, 6))
plt.scatter(odo, target)
plt.plot(odo, predicted_y, c = 'r')
plt.title('Scatter plot and a Simple Linear Regression Model')
plt.ylabel("price")
plt.xlabel("odo")
plt.show()

mse = cross_val_score(simple_lr,
                      odo,
                      target,
                      scoring='neg_mean_squared_error',
                      cv=10)

#print(mse.mean())
#joblib.dump(simple_lr, 'linear_ML_models/linear_regression_odo.pkl')

