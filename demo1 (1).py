import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('heart.csv')
print(data.head())
print(data.info())
print(data.isnull().sum())
dataset = data.copy()
X = dataset.drop(['target'], axis = 1)
print(X.head())
y = dataset['target']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=20)
model.fit(X_train, y_train)
pred = model.predict(X_test)

#print(pred[:10])
input=(45,1,3,150,178,1,0,100,0,2.3,0,0,1)
input_as_numpy=np.asarray(input)
input_reshaped=input_as_numpy.reshape(1,-1)
pre1=model.predict(input_reshaped)
print(pre1)
if(pre1==1): 
  print("The patient seems to be have heart disease")
else:
  print("The patient seems to be Normal")

print('logistic regression')

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()

classifier.fit(X_train, y_train)
pred = classifier.predict(X_test)

#print(pred[:10])
input=(45,1,3,150,178,1,0,100,0,2.3,0,0,1)
input_as_numpy=np.asarray(input)
input_reshaped=input_as_numpy.reshape(1,-1)
pre1=classifier.predict(input_reshaped)
print(pre1)
if(pre1==1): 
  print("The patient seems to be have heart disease")
else:
  print("The patient seems to be Normal")

print('svm classifier')
from sklearn.svm import SVC
classifier = SVC()
classifier.fit(X_train, y_train)
pred = classifier.predict(X_test)

#print(pred[:10])

input=(45,1,3,150,178,1,0,100,0,2.3,0,0,1)
input_as_numpy=np.asarray(input)
input_reshaped=input_as_numpy.reshape(1,-1)
pre1=classifier.predict(input_reshaped)
print(pre1)
if(pre1==1): 
  print("The patient seems to be have heart disease")
else:
  print("The patient seems to be Normal")




