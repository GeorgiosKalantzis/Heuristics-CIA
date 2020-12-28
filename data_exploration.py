
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


data = pd.read_csv('swipes.csv', header = 0)

# Move target playerID to the first column for easier manipulation
player = data['playerID']
data.drop(labels=['playerID'], axis=1,inplace = True)
data.insert(0, 'playerID', player)

data.drop(labels=['id'], axis=1,inplace = True)

# creating instance of labelencoder
labelencoder = LabelEncoder()

data['screen'] = labelencoder.fit_transform(data['screen'])
data['deviceOS'] = labelencoder.fit_transform(data['deviceOS'])
data['direction'] = labelencoder.fit_transform(data['direction'])
data['userGender'] = labelencoder.fit_transform(data['userGender'])

y = data.iloc[0:,0]
x = data.iloc[0:,1:]


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



