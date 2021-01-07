
"""
One-Class classification
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
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import OneClassSVM
import seaborn as sns



data = pd.read_csv('swipesNew.csv', header = 0)

# Move target playerID to the first column for easier manipulation
player = data['playerID']
data.drop(labels=['playerID'], axis=1,inplace = True)
data.insert(0, 'playerID', player)

# Drop swipe id
data.drop(labels=['id'], axis=1,inplace = True)


# Drop information on users with only one swipe

index1 = data[data['playerID'] == 'y2opfq8'].index
index2 = data[data['playerID'] == 'bh54lsq'].index
index3 = data[data['playerID'] == 'lvrishm'].index
index4 = data[data['playerID'] == '3cd5kys'].index

data.drop(index1 , inplace=True)
data.drop(index2 , inplace=True)
data.drop(index3 , inplace=True)
data.drop(index4 , inplace=True)

# Drop missing values and age which has missing values and it's useless
# poly_cof_determination is useless , no variance
data = data.drop('age', axis = 1)
data = data.drop('screen', axis = 1)
data = data.drop('deviceOS', axis = 1)
data = data.drop('userGender', axis = 1)
data = data.drop('userXP', axis = 1)
data = data.drop('poly_cof_determination', axis = 1)


# creating instance of labelencoder
labelencoder = LabelEncoder()

# Encode direction
data['direction'] = labelencoder.fit_transform(data['direction'])

# Drop rows with missing values
data = data.dropna()



y = data.iloc[0:,0].copy()
x = data.iloc[0:,1:]
# Standard scaling also for the encoded features
X = StandardScaler().fit_transform(x)
x = pd.DataFrame(X,index=x.index, columns = x.columns)

# All users name
users = y.value_counts().index

accuracies = []
FAR = []
FRR = []

# One-class classification for every user
for i in range(len(users)):

    print(users[i])
    
    y[y == users[i]] = 1
    
    y[y != 1] = -1
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state = 2020, stratify=y)   
    
    X_train_user = X_train[y_train == 1]
    X_train_attacker = X_train[y_train == -1]
    
    outlier_prop = len(X_train_user) / len(X_train_attacker)
    
    svm = OneClassSVM(kernel='rbf', nu=outlier_prop, gamma=0.000001) 
    # Train on user's data
    svm.fit(X_train_user)
    
    # Predict on all data
    pred = svm.predict(X_test)
    
    y_test = y_test.astype('int')
    
    accuracies.append(accuracy_score(y_test, pred))
    
    
    """
    # Compute False Acceptance Rate and False Rejection Rate
    CM = confusion_matrix(y_test,pred)

    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    
    FAR.append(FN/len(y_test[y_test == -1]))
    
    FRR.append(FP/len(y_test[y_test == 1]))
    """
    
    predA = svm.predict(X_test[y_test == -1])
    
    FAR.append(len(predA[predA == 1])/len(predA))
    
    predU = svm.predict(X_test[y_test == 1])
    
    FRR.append(len(predU[predU == -1])/len(predU))
    
    
    y = data.iloc[0:,0].copy()


print(np.mean(accuracies))
print(np.mean(FAR))
print(np.mean(FRR))









