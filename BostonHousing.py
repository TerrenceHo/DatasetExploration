from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

boston = load_boston()

#Linear Regression
model = LinearRegression()
model.fit(boston.data, boston.target)
expected = boston.target
predicted = model.predict(boston.data)

print("Linear regression model \n boston dataset") 
print("Mean squared error = %0.3f"%mse(expected, predicted))
print("R2 score = %0.3f"% r2_score(expected, predicted))

#Ridge Regression
model = Ridge(alpha = 0.1)
model.fit(boston.data, boston.target)
expected = boston.target
predicted = model.predict(boston.data)

print("Ridge regression model \n boston dataset")
print("Mean squared error = %0.3f" % mse(expected, predicted))
print("R2 score = %0.3f" % r2_score(expected, predicted))

#Random Forest
model = RandomForestRegressor()
model.fit(boston.data, boston.target)
expected = boston.target
predicted = model.predict(boston.data)

print("Random Forest model \n boston dataset")
print("Mean squared error = %0.3f" % mse(expected, predicted))
print("R2 score = %0.3f" % r2_score(expected, predicted))