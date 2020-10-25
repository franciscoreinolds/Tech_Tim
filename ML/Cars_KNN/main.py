import sklearn
from sklearn import linear_model, preprocessing
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np

# read data
data = pd.read_csv("car.data")

# change strings to numerical values
le = preprocessing.LabelEncoder()

# create lists of every attribute and converting them to numerical
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

predict = "class"

# formatting data
X = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)

# create training and testing datasets
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

# create classifier
model = KNeighborsClassifier(n_neighbors=5)

# train model
model.fit(x_train, y_train)

# check accuracy
acc = model.score(x_test, y_test)
print(acc)

# predict data
predicted = model.predict(x_test)

names = ["unacc", "acc", "good", "vgood"]

for x in range(len(predicted)):
    print("Predicted: ", names[predicted[x]], "Data: ", x_test[x], "Actual: ", names[y_test[x]])