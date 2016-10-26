from sklearn.datasets import load_diabetes
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from pylab import savefig

diabetes = load_diabetes()

# #Linear Regression
# model = LinearRegression()
# model.fit(diabetes.data, diabetes.target)
# expected = diabetes.target
# predicted = model.predict(diabetes.data)

# print("Linear regression model \n Diabetes dataset") 
# print("Mean squared error = %0.3f"%mse(expected, predicted))
# print("R2 score = %0.3f"% r2_score(expected, predicted))

#Linear Regression
model = LinearRegression()
diabetes_X = diabetes.data[:, np.newaxis, 2]
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

model.fit(diabetes_X_train, diabetes_y_train)

print('Linear Regression')
print('Coefficients: \n', model.coef_)
print("Mean squared error: %.2f"% np.mean((model.predict(diabetes_X_test) - diabetes_y_test) ** 2))
print('Variance score: %.2f' % model.score(diabetes_X_test, diabetes_y_test))

plt.scatter(diabetes_X_test, diabetes_y_test, color = 'black')
plt.plot(diabetes_X_test, model.predict(diabetes_X_test), color = 'blue', linewidth = 3)
plt.xticks(())
plt.yticks(())
savefig('diabetesLinearRegression.png')
plt.show()

model = Ridge(alpha = 0.1)
diabetes_X = diabetes.data[:, np.newaxis, 2]
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

model.fit(diabetes_X_train, diabetes_y_train)

#Ridge Regression
print('Ridge Regression')
print('Coefficients: \n', model.coef_)
print("Mean squared error: %.2f"% np.mean((model.predict(diabetes_X_test) - diabetes_y_test) ** 2))
print('Variance score: %.2f' % model.score(diabetes_X_test, diabetes_y_test))

plt.scatter(diabetes_X_test, diabetes_y_test, color = 'black')
plt.plot(diabetes_X_test, model.predict(diabetes_X_test), color = 'red', linewidth = 3)
plt.xticks(())
plt.yticks(())
savefig('diabetesRidgeRegression.png')
plt.show()

# model = RandomForestRegressor()
# diabetes_X = diabetes.data[:, np.newaxis, 2]
# diabetes_X_train = diabetes_X[:-20]
# diabetes_X_test = diabetes_X[-20:]

# diabetes_y_train = diabetes.target[:-20]
# diabetes_y_test = diabetes.target[-20:]

# model.fit(diabetes_X_train, diabetes_y_train)

# #print('Coefficients: \n', model.coef_)
# print("Mean squared error: %.2f"% np.mean((model.predict(diabetes_X_test) - diabetes_y_test) ** 2))
# print('Variance score: %.2f' % model.score(diabetes_X_test, diabetes_y_test))

# plt.scatter(diabetes_X_test, diabetes_y_test, color = 'black')
# plt.plot(diabetes_X_test, model.predict(diabetes_X_test), color = 'green', linewidth = 3)
# plt.xticks(())
# plt.yticks(())
# savefig('diabetesRandomForest.png')
# plt.show()

# #Ridge Regression
# model = Ridge(alpha = 0.1)
# model.fit(diabetes.data, diabetes.target)
# expected = diabetes.target
# predicted = model.predict(diabetes.data)

# print("Ridge regression model \n Diabetes dataset")
# print("Mean squared error = %0.3f" % mse(expected, predicted))
# print("R2 score = %0.3f" % r2_score(expected, predicted))

#Random Forest
model = RandomForestRegressor()
model.fit(diabetes.data, diabetes.target)
expected = diabetes.target
predicted = model.predict(diabetes.data)

print("Random Forest model \n Diabetes dataset")
print("Mean squared error = %0.3f" % mse(expected, predicted))
print("R2 score = %0.3f" % r2_score(expected, predicted))

