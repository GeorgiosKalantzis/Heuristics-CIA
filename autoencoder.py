import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
#from tensorflow.keras.losses import MeanSquaredLogarithmicError
from tensorflow import keras
from math import exp

 # create a model by subclassing Model class in tensorflow
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

data = data[data['screen'].str.contains('FocusGame.*')]

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



# All users name
users = y.value_counts().index
#Deleting users with few swipes
for i in range(len(users)-1,len(users)-51,-1):
    users=users.delete(i)
    
accuracies = []
FAR = []
FRR = []

attackers_accepted_swipes = []
SLR = []

#for i in range(len(users)):
    
y[y == users[10]] = 1

y[y != 1] = -1

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2020, stratify=y)

X_train_user = X_train[y_train == 1]
X_train_attacker = X_train[y_train == -1]

   
model = AutoEncoder(output_units=X_train_user.shape[1])
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
predictions = get_predictions(model, X_test.values, threshold)
y_test = y_test.astype('int')

#accuracies.append(accuracy_score(predictions, y_test))
print("Accuracy: ",accuracy_score(predictions, y_test))


CM = confusion_matrix(y_test, predictions)
TN = CM[0][0]
FN = CM[1][0]
TP = CM[1][1]
FP = CM[0][1]

#FAR.append(FP/(FP+TN))
#FRR.append(FN/(TP+FN))
print("FAR: ", FAR)
print("FRR: ", FRR)
    

## -- Authentication Proccess --------
predA = get_predictions(model, X_test[y_test == -1].values, threshold)
predU = get_predictions(model, X_test[y_test == 1].values, threshold)
 
confidence_level = 0.55
# Attackers accepted swipes until lock
swipes_till_lock = 0
locked = False
a = 0.2
b = 0.2
c = 0.2
# Decision sequence
D = [0.0 for _ in range(len(predA)+1)] 
# Initial value
D[0] = 0.55
status = 'fail'
for i in range(len(predA)):
 
    if(predA[i] == -1):
        #confidence_level = confidence_level + a*predA[i]
        
        D[i+1] = D[i] - (1-0.8*exp(-5*exp(-0.005*i)))*a
        
        if i>0:
            if D[i+1]<D[i] and D[i+1]<D[i-1]:
                D[i+1] = D[i+1] + exp(-i+1)*c 
            else:
                pass
        
        
    else:
        D[i+1] = D[i] + (1-0.8*exp(-5*exp(-0.005*i)))*b
        
        if i>0:
            if D[i+1]>D[i] and D[i+1]>D[i-1]:
                D[i+1] = D[i+1] - exp(-i+1)*c 
            else:
                pass
        
    if D[i+1] > 0.8:
        break
        
        
    if D[i+1]< 0.3:
        status = 'success'
        swipes_till_lock = i + 1
        break
    
    """
        #confidence_level = confidence_level + b*predA[i]
    if(confidence_level > 1):
        confidence_level = 1
    
    if(confidence_level < 0.38):
        locked = True
        swipes_till_lock = i + 1
        break;
    """

#if(not locked):
 #   swipes_till_lock = len(predA) + 1

print("Session of attacker: ", status)
print("Attacker Accepted swipes: ",swipes_till_lock)


# User rejection rate
#confidence_level = 0.55
count_locks = 0

# Decision sequence
D = [0.0 for _ in range(len(predA))] 
# Initial value
D[0] = 0.55
status = 'fail'


for i in range(len(predU)):
    
    if(predU[i] == -1):
        
        #confidence_level = confidence_level - a
        
        D[i+1] = D[i] - (1-0.8*exp(-5*exp(-0.005*i)))*a
        
        if i>0:
            if D[i+1]<D[i] and D[i+1]<D[i-1]:
                D[i+1] = D[i+1] + exp(-i+1)*c 
            else:
                pass
        
        
    else:
        #confidence_level = confidence_level + b
        D[i+1] = D[i] + (1-0.8*exp(-5*exp(-0.005*i)))*b
        
        if i>0:
            if D[i+1]>D[i] and D[i+1]>D[i-1]:
                D[i+1] = D[i+1] - exp(-i+1)*c 
            else:
                pass
            
    if D[i+1] > 0.8:
        status = 'success'
        break
        
        
    if D[i+1]< 0.3:
        break
        
        
    """
    if(confidence_level > 1):
        confidence_level = 1
        
    
    if(confidence_level < 0.38):

        count_locks += 1
        confidence_level = 0.55
    """

print("Session of user: ", status)
        
#SLR.append(count_locks/len(predU))
#print("SLR: ", SLR)
#y = data.iloc[0:,0].copy()

"""
print('Accuracy: ',np.mean(accuracies))
print('FAR: ',np.mean(FAR))
print('FRR: ',np.mean(FRR))
print('Attackers accepted swipes: ',np.mean(attackers_accepted_swipes))
print('SLR: ',np.mean(SLR))
"""





































