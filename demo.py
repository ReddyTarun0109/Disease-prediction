import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

df = pd.read_csv('Iris.csv')
print(df.head())
df = df.drop(columns = ['Id'])
#print(df.info())
#print(df.isnull().sum())

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df['Species'] = le.fit_transform(df['Species'])
#print(df.head())

from sklearn.model_selection import train_test_split
# train - 70
# test - 30
X = df.drop(columns=['Species'])
print(X.head())
Y = df['Species']
print(Y.head())
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train, y_train)
print("Accuracy LogisticRegression: ",model.score(x_test, y_test) * 100)
lr=model.score(x_test, y_test) * 100
input=(5.9,3.5,1.5,1.9)
input_as_numpy=np.asarray(input)
input_reshaped=input_as_numpy.reshape(1,-1)
pre1=model.predict(input_reshaped)
print(pre1)
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(x_train, y_train)
print("Accuracy KNN: ",model.score(x_test, y_test) * 100)
kn=model.score(x_test, y_test) * 100
input=(5.9,3.5,1.5,1.9)
input_as_numpy=np.asarray(input)
input_reshaped=input_as_numpy.reshape(1,-1)
pre1=model.predict(input_reshaped)
print(pre1)
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(x_train, y_train)
print("Accuracy: Decesion Tree ",model.score(x_test, y_test) * 100)
dt=model.score(x_test, y_test) * 100
input=(5.9,3.5,1.5,1.9)
input_as_numpy=np.asarray(input)
input_reshaped=input_as_numpy.reshape(1,-1)
pre1=model.predict(input_reshaped)
print(pre1)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=20)
model.fit(x_train, y_train)
print("Accuracy Random Forest: ",model.score(x_test, y_test) * 100)
rf=model.score(x_test, y_test) * 100
input=(5.9,3.5,1.5,1.9)
input_as_numpy=np.asarray(input)
input_reshaped=input_as_numpy.reshape(1,-1)
pre1=model.predict(input_reshaped)
print(pre1)
from sklearn.svm import SVC
classifier = SVC()
model.fit(x_train, y_train)
print("Accuracy SVM: ",model.score(x_test, y_test) * 100)
sv=model.score(x_test, y_test) * 100
input=(5.9,3.5,1.5,1.9)
input_as_numpy=np.asarray(input)
input_reshaped=input_as_numpy.reshape(1,-1)
pre1=model.predict(input_reshaped)
print(pre1)

import numpy as np
import matplotlib.pyplot as plt
data = {'lr':lr, 'knn':kn, 'dt':dt,'rf':rf,'svm':sv}
courses = list(data.keys())
values = list(data.values())
fig = plt.figure(figsize = (10, 5))
plt.bar(courses, values, color ='maroon',width = 0.4)
plt.xlabel("Alogritham")
plt.ylabel("accuries")
plt.title("comprasion graph")
plt.show()




