#import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplot
from matplotlib import style
import pickle
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle


#tf.config.experimental.list_physical_devices('GPU')

data = pd.read_csv("student-mat.csv", sep = ";")

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

best = 0

for it in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
    linear = linear_model.LinearRegression()

    # train model
    linear.fit(x_train, y_train)

    # score model
    acc = linear.score(x_test, y_test)

    if (acc > best):
        # print accuracy, coefficients and intercept point
        print("Accuracy: ", acc)
        
        best = acc
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)

# open model
pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

# score model
acc = linear.score(x_test, y_test)

# print accuracy, coefficients and intercept point
print("Final Accuracy: ", acc)

predictions = linear.predict(x_test)

print("Co: ", linear.coef_)
print("Intercept: ", linear.intercept_)

"""
# print predictions
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])
"""

# plot
p = 'G1'
style.use("ggplot")
pyplot.scatter(data[p], data['G3'])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()