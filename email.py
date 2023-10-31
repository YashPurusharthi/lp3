import pandas as pd
df = pd.read_csv('Dataset/emails.csv')
df.shape
df.head()

# Input Data
x = df.drop(['Email No.','Prediction'], axis = 1)

# Output Data
y = df['Prediction']
x.shape
x.dtypes
set(x.dtypes)
import seaborn as sns
sns.countplot(x = y);
y.value_counts()

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)
x_scaled

# Cross Validation
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, random_state = 0 , test_size = 0.25)
x_scaled.shape
x_train.shape
x_test.shape

# Import the class
from sklearn.neighbors import KNeighborsClassifier
# Create the object
knn = KNeighborsClassifier(n_neighbors=5)
# Train the algorithm
knn.fit(x_train, y_train)

# predict on test data
y_pred = knn.predict(x_test)
# import the evaluation metrics
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.metrics import classification_report
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)

y_test.value_counts()
accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))

import numpy as np
import matplotlib as plt
error = []
for k in range(1,41):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    pred = knn.predict(x_test)
    error.append(np.mean(pred != y_test))
error

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)
accuracy_score(y_test, y_pred)

from sklearn.svm import SVC
svm = SVC(kernel='poly')
svm.fit(x_train, y_train)

y_pred = svm.predict(x_test)
accuracy_score(y_test, y_pred)