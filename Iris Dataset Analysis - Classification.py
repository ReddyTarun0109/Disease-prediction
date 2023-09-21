

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

df = pd.read_csv('Iris.csv')
print(df.head())


df = df.drop(columns = ['Id'])
print(df.head())

print(df.describe())

print(df.info())

print(df.isnull().sum())




# # Label Encoder
# 
# In machine learning, we usually deal with datasets which contains multiple labels in one or more than one columns. These labels can be in the form of words or numbers. Label Encoding refers to converting the labels into numeric form so as to convert it into the machine-readable form

# In[19]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[20]:


df['Species'] = le.fit_transform(df['Species'])
print(df.head())




from sklearn.model_selection import train_test_split
# train - 70
# test - 30
X = df.drop(columns=['Species'])
Y = df['Species']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)



# logistic regression 
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


model.fit(x_train, y_train)


print("Accuracy: ",model.score(x_test, y_test) * 100)
# knn - k-nearest neighbours
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()

model.fit(x_train, y_train)

# print metric to get performance
print("Accuracy: ",model.score(x_test, y_test) * 100)
# decision tree
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()

model.fit(x_train, y_train)
print("Accuracy: ",model.score(x_test, y_test) * 100)


