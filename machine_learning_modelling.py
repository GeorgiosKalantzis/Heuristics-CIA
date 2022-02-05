import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from math import exp
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
import math
import random
from sympy import symbols,plot
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow import keras
class AutoEncoder(Model):
  """
  Parameters
  ----------
  output_units: int
    Number of output units
  
  code_size: int
    Number of units in bottle neck
  """

  def __init__(self, output_units, code_size=6):
    super().__init__()
    
    self.encoder = Sequential([
      Dense(15, activation='tanh'),
     
      Dense(code_size, activation='tanh')
    ])
    self.decoder = Sequential([
     
      Dense(15, activation='tanh'),
      Dense(output_units, activation='sigmoid')
    ])
  
  def call(self, inputs):
    encoded = self.encoder(inputs)
    decoded = self.decoder(encoded)
    return decoded
    
def get_errors(model, x_test_scaled, threshold):
  predictions = model.predict(x_test_scaled)
  # provides losses of individual instances
  #errors = tf.keras.losses.msle(predictions, x_test_scaled)
  errors = np.mean(np.square(np.log(predictions + 1) - np.log(x_test_scaled + 1)), axis=-1)
  return errors


def find_threshold(model, x_train_scaled):
  reconstructions = model.predict(x_train_scaled)
  # provides losses of individual instances
  #reconstruction_errors = tf.keras.losses.mean_squared_logarithmic_error(reconstructions, x_train_scaled)
  reconstruction_errors = np.mean(np.square(np.log(reconstructions + 1) - np.log(x_train_scaled + 1)), axis=-1)
  # threshold for anomaly scores
  threshold = np.mean(reconstruction_errors \
      + np.std(reconstruction_errors))
  return threshold

def get_predictions(model, x_test_scaled, threshold):
  predictions = model.predict(x_test_scaled)
  # provides losses of individual instances
  #errors = tf.keras.losses.msle(predictions, x_test_scaled)
  errors = np.mean(np.square(np.log(predictions + 1) - np.log(x_test_scaled + 1)), axis=-1)
  # 0 = anomaly, 1 = normal
  
  anomaly_mask = pd.Series(errors) > threshold
  preds = anomaly_mask.map(lambda x: -1 if x == True else 1)
  return preds


def load_user_data(user_id):
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
    #data = data.drop('screen', axis=1)
    data = data.drop('deviceOS', axis=1)
    data = data.drop('userGender', axis=1)
    data = data.drop('userXP', axis=1)
    data = data.drop('poly_cof_determination', axis=1)
    data = data.drop('deviceHeight', axis = 1)
    data = data.drop('deviceWidth', axis = 1)
    data = data.drop('traceMedianAbsoluteError', axis = 1)
    data = data.drop('traceMeanAbsoluteError', axis = 1)
    data = data.drop('direction', axis = 1)
    
    data = data[data['screen'].str.contains('MathisisGame.*')]
    
    data = data.drop('screen', axis = 1)
    
    
    # Drop rows with missing values
    data = data.dropna()
    
    y = data.iloc[0:, 0].copy()
    x = data.iloc[0:, 1:]
    # Standard scaling also for the encoded features
    
    
    
    min_max_scaler = MinMaxScaler(feature_range=(0, 1))
    #X = StandardScaler().fit_transform(x)
    X = min_max_scaler.fit_transform(x)
    x = pd.DataFrame(X, index=x.index, columns=x.columns)
    users = y.value_counts().index
    #Deleting users with few swipes
    for i in range(len(users)-1,len(users)-51,-1):
        users=users.delete(i)
        
            
    y[y == users[user_id]] = 1
    
    y[y != 1] = -1
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2020, stratify=y)
    
    X_train_user = X_train[y_train == 1]
    X_train_attacker = X_train[y_train == -1]
    
    return X_train_user,X_train_attacker,X_test,y_test

def classification(X_train_user,X_train_attacker,X_test,y_test,classifier):
    predictions = np.array([0 for x in range(len(y_test))])
    user_predictions = np.array([0 for x in range(len(y_test[y_test == 1]))])
    attacker_predictions = np.array([0 for x in range(len(y_test[y_test == -1]))])

    if classifier=='LOF' :
        model= LocalOutlierFactor(n_neighbors = 1, novelty = True)
        model.fit(X_train_user)
        
        predictions=model.predict(X_test)
        user_predictions=model.predict(X_test[y_test==1])
        attacker_predictions=model.predict(X_test[y_test==-1])
        
    elif classifier=='Autoencoder':
        model = AutoEncoder(output_units=X_train_user.shape[1])
        y_test = y_test.astype('int')
    
    # configurations of model
        model.compile(loss='msle', metrics=['mse'], optimizer='adam')
        history = model.fit(
            X_train_user.values,
            X_train_user.values,
            epochs=20,
            batch_size=512,
            validation_data=(X_test.values, X_test.values)
        )
        threshold = find_threshold(model, X_train_user.values)
        predictions= get_predictions(model, X_test.values, threshold)
        y_test = y_test.astype('int')
        attacker_predictions = get_predictions(model, X_test[y_test == -1].values, threshold)
        user_predictions = get_predictions(model, X_test[y_test == 1].values, threshold)
    else:
        init_pred=np.array([[0 for x in range(len(y_test))]for y in range(7)])
        user_init_pred=np.array([[0 for x in range(len(y_test[y_test==1]))]for y in range(7)])
        attacker_init_pred=np.array([[0 for x in range(len(y_test[y_test==-1]))]for y in range(7)])
        
        outlier_prop = len(X_train_user) / len(X_train_attacker)
        
        # Fit the model and also save them for later use
        lof1 = LocalOutlierFactor(n_neighbors = 1, novelty = True)
        lof1.fit(X_train_user)
        
     
        
        lof2 = LocalOutlierFactor(n_neighbors = 2, novelty = True)
        lof2.fit(X_train_user)
        
        
        lof3 = LocalOutlierFactor(n_neighbors = 3, novelty = True)
        lof3.fit(X_train_user)
        
        
        
        lof4 = LocalOutlierFactor(n_neighbors = 4, novelty = True)
        lof4.fit(X_train_user)
        
        
        svm1 = OneClassSVM(kernel='rbf', nu=outlier_prop, gamma=0.1)
        svm1.fit(X_train_user)
        
        svm2 = OneClassSVM(kernel='rbf', nu=outlier_prop, gamma=0.5)
        svm2.fit(X_train_user)
        
        
        svm3 = OneClassSVM(kernel='rbf', nu=outlier_prop, gamma=0.9)
        svm3.fit(X_train_user)
        
        
        init_pred[0,:] = lof1.predict(X_test)
        init_pred[1,:] = lof2.predict(X_test)
        init_pred[2,:] = lof3.predict(X_test)
        init_pred[3,:] = lof4.predict(X_test)
        init_pred[4,:] = svm1.predict(X_test)
        init_pred[5,:] = svm2.predict(X_test)
        init_pred[6,:] = svm3.predict(X_test)
       
        
        
        
        user_init_pred[0,:] = lof1.predict(X_test[y_test==1])
        user_init_pred[1,:] = lof2.predict(X_test[y_test==1])
        user_init_pred[2,:] = lof3.predict(X_test[y_test==1])
        user_init_pred[3,:] = lof4.predict(X_test[y_test==1])
        user_init_pred[4,:] = svm1.predict(X_test[y_test==1])
        user_init_pred[5,:] = svm2.predict(X_test[y_test==1])
        user_init_pred[6,:] = svm3.predict(X_test[y_test==1])
        
        
        
        
        attacker_init_pred[0,:] = lof1.predict(X_test[y_test==-1])
        attacker_init_pred[1,:] = lof2.predict(X_test[y_test==-1])
        attacker_init_pred[2,:] = lof3.predict(X_test[y_test==-1])
        attacker_init_pred[3,:] = lof4.predict(X_test[y_test==-1])
        attacker_init_pred[4,:] = svm1.predict(X_test[y_test==-1])
        attacker_init_pred[5,:] = svm2.predict(X_test[y_test==-1])
        attacker_init_pred[6,:] = svm3.predict(X_test[y_test==-1])
      
        
        predictions = np.array([0 for x in range(len(y_test))])
        user_predictions = np.array([0 for x in range(len(y_test[y_test == 1]))])
        attacker_predictions = np.array([0 for x in range(len(y_test[y_test == -1]))])
        
        # Voting
        for w in range(len(y_test)):
                class_votes=np.array([0 for x in range(2)])
                
                for z in range(7):
                    if init_pred[z,w]==-1:
                        class_votes[0]=class_votes[0]+1
                    else:
                        class_votes[1]=class_votes[1]+1
                    
        
                index_max=np.argmax(class_votes)
                
    
                if index_max==0:
                    predictions[w]=-1
                else:
                    predictions[w]=1
            
                
        for w in range(len(y_test[y_test==1])):
                
                class_votes1=np.array([0 for x in range(2)])
                
                for z in range(7):
                    if user_init_pred[z,w]==-1:
                        class_votes1[0]=class_votes1[0]+1
                    else:
                        class_votes1[1]=class_votes1[1]+1
                    
                
                        
                
                index_max1=np.argmax(class_votes1)
                
                if index_max1==0:
                    user_predictions[w]=-1
                else:
                    user_predictions[w]=1
                    
                    
        for w in range(len(y_test[y_test==-1])):
                
            class_votes2=np.array([0 for x in range(2)])
            
            for z in range(7):
                if attacker_init_pred[z,w]==-1:
                    class_votes2[0]=class_votes2[0]+1
                else:
                    class_votes2[1]=class_votes2[1]+1
                
            
                    
            
            index_max2=np.argmax(class_votes2)
            
            if index_max2==0:
                attacker_predictions[w]=-1
            else:
                attacker_predictions[w]=1
                
    return predictions,user_predictions,attacker_predictions

def metrics(predictions,y_test):
    y_test = y_test.astype('int')
    CM = confusion_matrix(y_test, predictions)
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    FAR=FP/(TN+FP)
    FRR=FN/(TP+FN)
    return FAR,FRR


    

