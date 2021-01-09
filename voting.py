"""
One-Class classification
"""
from sklearn.metrics import precision_score
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import OneClassSVM

data = pd.read_csv('swipesNew.csv', header=0)

# Move target playerID to the first column for easier manipulation
player = data['playerID']
data.drop(labels=['playerID'], axis=1, inplace=True)
data.insert(0, 'playerID', player)

# Drop swipe id
data.drop(labels=['id'], axis=1, inplace=True)

# Drop information on users with only one swipe

index1 = data[data['playerID'] == 'y2opfq8'].index
index2 = data[data['playerID'] == 'bh54lsq'].index
index3 = data[data['playerID'] == 'lvrishm'].index
index4 = data[data['playerID'] == '3cd5kys'].index

data.drop(index1, inplace=True)
data.drop(index2, inplace=True)
data.drop(index3, inplace=True)
data.drop(index4, inplace=True)

# Drop missing values and age which has missing values and it's useless
# poly_cof_determination is useless , no variance
data = data.drop('age', axis=1)
data = data.drop('screen', axis=1)
data = data.drop('deviceOS', axis=1)
data = data.drop('userGender', axis=1)
data = data.drop('userXP', axis=1)
data = data.drop('poly_cof_determination', axis=1)

# creating instance of labelencoder
labelencoder = LabelEncoder()

# Encode direction
data['direction'] = labelencoder.fit_transform(data['direction'])

# Drop rows with missing values
data = data.dropna()

y = data.iloc[0:, 0].copy()
x = data.iloc[0:, 1:]
# Standard scaling also for the encoded features
X = StandardScaler().fit_transform(x)
x = pd.DataFrame(X, index=x.index, columns=x.columns)

# All users name
users = y.value_counts().index
for i in range(969,919,-1):
    users=users.delete(i)
accuracies = []
FAR = []
FRR = []
precision=[]
predictions=np.array([[0 for x in range(43274)]for y in range(5)])
# One-class classification for every user

for i in range(len(users)):
    #print(users[i])
    print(i)
    y[y == users[i]] = 1

    y[y != 1] = -1

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2020, stratify=y)

    X_train_user = X_train[y_train == 1]
    X_train_attacker = X_train[y_train == -1]

    outlier_prop = len(X_train_user) / len(X_train_attacker)

    svm1 = OneClassSVM(kernel='rbf', nu=outlier_prop, gamma=0.0001)
    svm2 = OneClassSVM(kernel='rbf', nu=outlier_prop, gamma=0.001)
    svm3 = OneClassSVM(kernel='rbf', nu=outlier_prop, gamma=0.01)
    svm4 = OneClassSVM(kernel='rbf', nu=outlier_prop, gamma=0.1)
    svm5 = OneClassSVM(kernel='rbf', nu=outlier_prop, gamma=1)

    # Train on user's data
    svm1.fit(X_train_user)
    svm2.fit(X_train_user)
    svm3.fit(X_train_user)
    svm4.fit(X_train_user)
    svm5.fit(X_train_user)

    # Predict on all data
    predictions[0, :] = svm1.predict(X_test)
    predictions[1, :] = svm2.predict(X_test)
    predictions[2, :] = svm3.predict(X_test)
    predictions[3, :] = svm4.predict(X_test)
    predictions[4, :] = svm5.predict(X_test)

    y_test = y_test.astype('int')
    #Combining the Decisions of models
    final_prediction = np.array([0 for x in range(len(y_test))])
    for w in range(len(y_test)):
            class_votes=np.array([0 for x in range(2)])
            for z in range(5):
                if predictions[z,w]==-1:
                    class_votes[0]=class_votes[0]+1
                else:
                    class_votes[1]=class_votes[1]+1
            index_max=np.argmax(class_votes)

            if index_max==0:
                final_prediction[w]=-1
            else:
                final_prediction[w]=1


    CM = confusion_matrix(y_test, final_prediction)
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    """
        for w in range(len(y_test)):

            if final_prediction[w] == 1 and y_test.values[w] == -1:
                FP = FP + 1
            if final_prediction[w] == -1 and y_test.values[w] == 1:
                FN = FN + 1
            if final_prediction[w]==-1 and y_test.values[w]==-1:
                TN=TN+1
            if final_prediction[w]==1 and y_test.values[w]==1:
                TP=TP+1
    """

    accuracies.append(accuracy_score(y_test, final_prediction))
    #print(final_prediction)
    # Compute False Acceptance Rate and False Rejection Rate

    FAR.append(FP/(FP+TN))
    FRR.append(FN/(TP+FN))
    y = data.iloc[0:, 0].copy()


print(np.mean(accuracies))
print(np.mean(FAR))
print(np.mean(FRR))
#print(np.std(FAR))
#print(np.std(FRR))


