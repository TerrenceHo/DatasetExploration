import os
import requests
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation as cv
from sklearn.metrics import classification_report
from sklearn.svm import SVC

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data"
response = requests.get(URL)
outpath = os.path.abspath("haberman.txt")
with open(outpath, 'wb') as f:
	f.write(response.content)

df = pd.read_csv("haberman.txt", header = None, names = ["age_at_op","op_yr","nr_nodes","survival"])
FEATURES = df[["age_at_op","op_yr","nr_nodes"]]
TARGETS = df[["survival"]]

splits = cv.train_test_split(FEATURES, TARGETS, test_size = 0.2)
X_train, X_test, y_train, y_test = splits

model = SVC()
model.fit(X_train, y_train.ravel())
expected = y_test
predicted = model.predict(X_test)
print("Support Vector Machine Classifier \n Haberman survival dataset")
print(classification_report(expected, predicted, target_names=[">=5 years","<5 years"]))

# model = LogisticRegression()
# model.fit(X_train, y_train.ravel())
# expected = y_test
# predicted = model.predict(X_test)

# print("Logistic Regression Classifier \n Haberman survival dataset")
# print(classification_report(expected, predicted, target_names = [">=5 years", "<5 years"])) 