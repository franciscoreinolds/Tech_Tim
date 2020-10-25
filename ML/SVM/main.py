import sklearn
from sklearn import datasets
from sklearn import svm

# Support Vector Machines

# Load dataset
cancer = datasets.load_breast_cancer()

print(cancer.feature_names)
print(cancer.target_names)

x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.15)

classes = ['malignant', 'benign']

# Classifier
clf = svm.SVC(kernel="linear")

# Fit classifier
clf.fit(x_train, y_train)

# Make predictions
y_pred = clf.predict(x_test)

# Check accuracy
acc = sklearn.metrics.accuracy_score(y_test, y_pred)

print(acc)
