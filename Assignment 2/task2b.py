import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from pylab import savefig

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import PolynomialFeatures

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

world = pd.read_csv("world.csv")
life = pd.read_csv('life.csv')

# Start cleaning
for index, row in world.iterrows():
    if not (row['Country Code'] in list(life['Country Code'])):
        world.drop(index, inplace=True)

# Merge values
features = world.columns[3:]
to_merge = life[['Country Code', 'Life expectancy at birth (years)']]
world = world.merge(to_merge, on='Country Code')

# Setting y variable after merge to get the right order of target var
y = world['Life expectancy at birth (years)']

# Dropping unnecessary columns
world_link = world[['Country Name', 'Time', 'Country Code']]
world.drop(['Country Name', 'Time', 'Country Code', 'Life expectancy at birth (years)'], axis = 1, inplace=True)
world.replace(["..", np.nan], inplace=True)

# New values is imputed world
imp_median = SimpleImputer(missing_values=np.nan, strategy="median")
imp_median.fit(world)
new_values = pd.DataFrame(imp_median.transform(world))
new_values.columns = features

# Generating new set of features (210 feats)
poly = PolynomialFeatures(2, interaction_only=True, include_bias = False)
new_features = poly.fit_transform(new_values)

print("Number of new features (without kmeans): " + str(len(new_features[0])))
# Assign column names
new_feat_df = pd.DataFrame(new_features)
new_feat_names = []
for i in range(1,21):
    for j in range(i,21):
        if not i == j:
            new_feat_names.append("f"+str(i)+"xf"+str(j))
new_feat_df.columns = list(features) + new_feat_names

accuracy = {'FE':[], 'PCA':[], 'F4':[]}
# For loop to find best number of cluster for KMeans
for i in range(1,21):
    
    # KMeans on the 'world'(new_values) dataset not 210 features
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(new_values)
    new_feat_df['kmeans'] = kmeans.labels_
    # New_feat_df is the dataframe that holds 211 features and to be used in the 3 cases

    # Scaler for all columns
    scaler = StandardScaler()
    scaled = pd.DataFrame(scaler.fit_transform(new_feat_df))
    scaled.columns = new_feat_df.columns


    # ----- Case 1: Using Feature Engineering
    sel = SelectFromModel(RandomForestClassifier(n_estimators=250, random_state=100), max_features=4)
    sel.fit(scaled, y)
    # Selected columns
    sel_cols = new_feat_df.columns[(sel.get_support())]

    X = scaled[sel_cols]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=100)

    knn5_FE = KNeighborsClassifier(n_neighbors=5)
    knn5_FE.fit(X_train, y_train)
    pred_FE = knn5_FE.predict(X_test)


    # ------ Case 2: PCA
    scalerPCA = StandardScaler()
    scaledPCA = pd.DataFrame(scalerPCA.fit_transform(new_values))
    scaledPCA.columns = new_values.columns
    
    pca = PCA(n_components=4, random_state=100).fit_transform(scaledPCA)
    # Since kNN requires scaled data 
    # X = pd.DataFrame(StandardScaler().fit_transform(pca))
    X_train, X_test, y_train, y_test = train_test_split(pca, y, test_size=1/3, random_state=100)
    
    knn5_PCA = KNeighborsClassifier(n_neighbors=5)
    knn5_PCA.fit(X_train, y_train)
    pred_PCA = knn5_PCA.predict(X_test)


    # ----- Case 3: First 4 features
    X = scaled.iloc[:,:4]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=100)

    knn5_F4 = KNeighborsClassifier(n_neighbors=5)
    knn5_F4.fit(X_train, y_train)
    pred_F4 = knn5_F4.predict(X_test)
    
    # For plot for to find ideal number of clusters
    accuracy['FE'].append(accuracy_score(y_test, pred_FE))
    accuracy['PCA'].append(accuracy_score(y_test, pred_PCA))
    accuracy['F4'].append(accuracy_score(y_test, pred_F4))

accuracy_pd = pd.DataFrame(accuracy, index=range(1,21))
# PCA is unchanged as seen
# print(accuracy_pd)

fig = sns.lineplot(data=accuracy_pd, palette='winter', color='green', dashes=False)
fig.set(title = 'Accuracy vs. KMeans clusters', xlabel = 'Number of clusters', ylabel='Accuracy of models')
fig.get_figure().savefig('task2bgraph1.png')

# Choosing max accuracy from PCA
max_index = 3         # Defualt as 3
max_score = 0
for index, row in accuracy_pd.iterrows():
    if row.sum() > max_score:
        # Index starts from 0 in df
        max_index = index
        max_score = row.sum()

print("Index ideal for clustering is : " + str(max_index))
# KMeans on the 'world'(new_values) dataset not 210 features
kmeans = KMeans(n_clusters=max_index)
kmeans.fit(new_values)
new_feat_df['kmeans'] = kmeans.labels_
print("All 211 features - ")
print(new_feat_df)
# New_feat_df is the dataframe that holds 211 features and to be used in the 3 cases

# Scaler for all columns
scaler = StandardScaler()
scaled = pd.DataFrame(scaler.fit_transform(new_feat_df))
scaled.columns = new_feat_df.columns


# ----- Case 1: Using Feature Engineering
sel = SelectFromModel(RandomForestClassifier(n_estimators=250, random_state=100), max_features=4)
sel.fit(scaled, y)
# Selected columns
print("The following selected columns from RandomForestClassifier for Feature Engineering from 211 total columns")
sel_cols = new_feat_df.columns[(sel.get_support())]
print(sel_cols)

X = scaled[sel_cols]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=100)

knn5_FE = KNeighborsClassifier(n_neighbors=5)
knn5_FE.fit(X_train, y_train)
pred_FE = knn5_FE.predict(X_test)


# ------ Case 2: PCA
scalerPCA = StandardScaler()
scaledPCA = pd.DataFrame(scalerPCA.fit_transform(new_values))
scaledPCA.columns = new_values.columns
    
pca = PCA(n_components=4, random_state=100).fit_transform(scaledPCA)
print("PCA values")
print(pca)
# Since kNN requires scaled data 
# X = pd.DataFrame(StandardScaler().fit_transform(pca))
X_train, X_test, y_train, y_test = train_test_split(pca, y, test_size=1/3, random_state=100)
    
knn5_PCA = KNeighborsClassifier(n_neighbors=5)
knn5_PCA.fit(X_train, y_train)
pred_PCA = knn5_PCA.predict(X_test)


# ----- Case 3: First 4 features
X = scaled.iloc[:,:4]
print("First 4 features")
print(X.head())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=100)

knn5_F4 = KNeighborsClassifier(n_neighbors=5)
knn5_F4.fit(X_train, y_train)
pred_F4 = knn5_F4.predict(X_test)

print("Accuracy of feature engineering: "+str(round(accuracy_score(y_test, pred_FE), 3)))
print("Accuracy of PCA: " + str(round(accuracy_score(y_test, pred_PCA), 3)))
print("Accuracy of first four features: " + str(round(accuracy_score(y_test, pred_F4), 3)))
