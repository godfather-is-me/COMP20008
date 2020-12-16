import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score

world = pd.read_csv("world.csv")
life = pd.read_csv('life.csv')

# Drop columns not in life
for index, row in world.iterrows():
    if not (row['Country Code'] in list(life['Country Code'])):
        world.drop(index, inplace=True)

# Merge to get the right link
to_merge = life[['Country Code', 'Life expectancy at birth (years)']]
world = world.merge(to_merge, on='Country Code')

# Setting y variable after merge to get the right order of result
y = world['Life expectancy at birth (years)']
world.drop(['Life expectancy at birth (years)'], inplace=True, axis =1)

features = world.columns[3:]
world_link = world[['Country Name', 'Time', 'Country Code']]
world.drop(['Country Name', 'Time', 'Country Code'], axis = 1, inplace=True)
world.replace(["..", np.nan], inplace=True)

X = world
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=100)

# Split data -> Scale the data (used Std Scaler) -> Train -> Test
# Impute values
imp_median_train = SimpleImputer(missing_values=np.nan, strategy="median")
imp_median_test = SimpleImputer(missing_values=np.nan, strategy="median")
imp_median_train.fit(X_train)
imp_median_test.fit(X_test)

X_train = pd.DataFrame(imp_median_train.transform(X_train))
X_train.columns = features
X_test = pd.DataFrame(imp_median_test.transform(X_test))
X_test.columns = features

impute_mean = pd.DataFrame(X_train.mean())
impute_median = pd.DataFrame(X_train.median())
impute_var = pd.DataFrame(X_train.var())

# Task 2A has mean, median, and variance
task2a = pd.concat([impute_median, impute_mean, impute_var], axis = 1, sort=False)
task2a.columns = ['median', 'mean', 'variance']
task2a.index.name = 'feature'
task2a.to_csv("task2a.csv",float_format='%.3f')

scaler_train = StandardScaler()
scaler_test = StandardScaler()
scaler_train.fit(X_train)
scaler_test.fit(X_test)

X_train = pd.DataFrame(scaler_train.transform(X_train))
X_test = pd.DataFrame(scaler_test.transform(X_test))

# Dtree classifier
dtree = DecisionTreeClassifier(max_depth=4)
dtree.fit(X_train, y_train)
pred = dtree.predict(X_test)

# kNN classifier
knn5 = KNeighborsClassifier(n_neighbors=5)
knn5.fit(X_train, y_train)
pred5 = knn5.predict(X_test)

knn10 = KNeighborsClassifier(n_neighbors=10)
knn10.fit(X_train, y_train)
pred10 = knn10.predict(X_test)

#print(task2a)
print("Accuracy of decision tree: " + str(round(accuracy_score(y_test, pred), 3)))
print("Accuracy of k-nn (k=5): " + str(round(accuracy_score(y_test, pred5), 3)))
print("Accuracy of k-nn (k=10): " + str(round(accuracy_score(y_test, pred10), 3)))