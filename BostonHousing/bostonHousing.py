from sklearn.datasets import load_boston
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from pylab import savefig

boston = load_boston()

# #Linear Regression
# model = LinearRegression()
# model.fit(boston.data, boston.target)
# expected = boston.target
# predicted = model.predict(boston.data)

# print("Linear regression model \n boston dataset") 
# print("Mean squared error = %0.3f"%mse(expected, predicted))
# print("R2 score = %0.3f"% r2_score(expected, predicted))

#Linear Regression
model = LinearRegression()
boston_X = boston.data[:, np.newaxis, 2]
boston_X_train = boston_X[:-20]
boston_X_test = boston_X[-20:]

boston_y_train = boston.target[:-20]
boston_y_test = boston.target[-20:]

model.fit(boston_X_train, boston_y_train)

print('Linear Regression')
print('Coefficients: \n', model.coef_)
print("Mean squared error: %.2f"% np.mean((model.predict(boston_X_test) - boston_y_test) ** 2))
print('Variance score: %.2f' % model.score(boston_X_test, boston_y_test))

plt.scatter(boston_X_test, boston_y_test, color = 'black')
plt.plot(boston_X_test, model.predict(boston_X_test), color = 'blue', linewidth = 3)
plt.xticks(())
plt.yticks(())
savefig('bostonLinearRegression.png')
plt.show()

model = Ridge(alpha = 0.1)
boston_X = boston.data[:, np.newaxis, 2]
boston_X_train = boston_X[:-20]
boston_X_test = boston_X[-20:]

boston_y_train = boston.target[:-20]
boston_y_test = boston.target[-20:]

model.fit(boston_X_train, boston_y_train)

#Ridge Regression
print('Ridge Regression')
print('Coefficients: \n', model.coef_)
print("Mean squared error: %.2f"% np.mean((model.predict(boston_X_test) - boston_y_test) ** 2))
print('Variance score: %.2f' % model.score(boston_X_test, boston_y_test))

plt.scatter(boston_X_test, boston_y_test, color = 'black')
plt.plot(boston_X_test, model.predict(boston_X_test), color = 'red', linewidth = 3)
plt.xticks(())
plt.yticks(())
savefig('bostonRidgeRegression.png')
plt.show()

# model = RandomForestRegressor()
# boston_X = boston.data[:, np.newaxis, 2]
# boston_X_train = boston_X[:-20]
# boston_X_test = boston_X[-20:]

# boston_y_train = boston.target[:-20]
# boston_y_test = boston.target[-20:]

# model.fit(boston_X_train, boston_y_train)

# #print('Coefficients: \n', model.coef_)
# print("Mean squared error: %.2f"% np.mean((model.predict(boston_X_test) - boston_y_test) ** 2))
# print('Variance score: %.2f' % model.score(boston_X_test, boston_y_test))

# plt.scatter(boston_X_test, boston_y_test, color = 'black')
# plt.plot(boston_X_test, model.predict(boston_X_test), color = 'green', linewidth = 3)
# plt.xticks(())
# plt.yticks(())
# savefig('bostonRandomForest.png')
# plt.show()

# #Ridge Regression
# model = Ridge(alpha = 0.1)
# model.fit(boston.data, boston.target)
# expected = boston.target
# predicted = model.predict(boston.data)

# print("Ridge regression model \n boston dataset")
# print("Mean squared error = %0.3f" % mse(expected, predicted))
# print("R2 score = %0.3f" % r2_score(expected, predicted))

#Random Forest
model = RandomForestRegressor()
model.fit(boston.data, boston.target)
expected = boston.target
predicted = model.predict(boston.data)

print("Random Forest model \n boston dataset")
print("Mean squared error = %0.3f" % mse(expected, predicted))
print("R2 score = %0.3f" % r2_score(expected, predicted))
