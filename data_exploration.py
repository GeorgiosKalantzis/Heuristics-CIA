
"""
Data exploration
"""
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.cluster import DBSCAN


data = pd.read_csv('swipes.csv', header = 0)

# Move target playerID to the first column for easier manipulation
player = data['playerID']
data.drop(labels=['playerID'], axis=1,inplace = True)
data.insert(0, 'playerID', player)

# Drop swipe id
data.drop(labels=['id'], axis=1,inplace = True)


# Drop information on users with only one swipe
# Here oversampling is needed for these users
index1 = data[data['playerID'] == 'y2opfq8'].index
index2 = data[data['playerID'] == 'bh54lsq'].index
index3 = data[data['playerID'] == 'lvrishm'].index

data.drop(index1 , inplace=True)
data.drop(index2 , inplace=True)
data.drop(index3 , inplace=True)


# creating instance of labelencoder
labelencoder = LabelEncoder()

# Encoders for these features, in the future probably we will drop screen , deviceOS, userGender but not direction
data['screen'] = labelencoder.fit_transform(data['screen'])
data['deviceOS'] = labelencoder.fit_transform(data['deviceOS'])
data['direction'] = labelencoder.fit_transform(data['direction'])
data['userGender'] = labelencoder.fit_transform(data['userGender'])

y = data.iloc[0:,0]
x = data.iloc[0:,1:]
# Standard scaling also for the encoded features
x = StandardScaler().fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state = 2020, stratify=y)   

# One versus All svm classifier
clf = OneVsRestClassifier(SVC(),n_jobs=-1).fit(X_train, y_train)
prediction = clf.predict(X_test)

"""
# Entropy with respect to tagrget
importances = mutual_info_classif(x,y)
feat_importances = pd.Series(importances, x.columns[0:len(x.columns)])
feat_importances.plot(kind='barh', color='blue') #plot info gain of each features
plt.show()

# select the number of features you want to retain.
select_k = 10 #whatever we want

# create the SelectKBest with the mutual info(info gain) strategy.
selection = SelectKBest(mutual_info_classif, k=select_k).fit(x, y)

#plot the scores
plt.bar([i for i in range(len(selection.scores_))], selection.scores_)
plt.show()

# display the retained features.
features = x.columns[selection.get_support()]
print(features)
"""


