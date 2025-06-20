import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

sonar_data = pd.read_csv('sonar.csv')
# print(sonar_data.head())
# print(sonar_data.shape)
# print(sonar_data.describe())
# print(sonar_data.columns)

# print(sonar_data['R'].value_counts())

# print(sonar_data.groupby('R').mean())

X = sonar_data.drop(columns='R' , axis=1)
Y = sonar_data['R']

# print(X)
# print(Y)

# TRAINING AND TESTING

X_train,X_test,Y_train , Y_test = train_test_split(X,Y , test_size=0.1,stratify=Y , random_state=1)
# stratify is used to get almost equal numbers of rocks and mines in training data

# print(X.shape , X_test.shape , X_train.shape)
# print(X_train)
# print(Y_train)

# TRAINING THE MODEL
model = LogisticRegression()
model.fit(X_train , Y_train)

#ACCURACY OF THE MODEL ON TRAINING DATA
X_train_prediction = model.predict(X_train)
traning_data_accuracy= accuracy_score(X_train_prediction,Y_train)
#
# print(traning_data_accuracy)

#ACCURACY ON TEST DATA
X_test_prediction = model.predict(X_test )
test_data_accuracy= accuracy_score(X_test_prediction,Y_test)
#
# print(test_data_accuracy)


# MAKING A PREDICTIVE SYSTEM
input_data = (0.1313,0.2339,0.3059,0.4264,0.4010,0.1791,0.1853,0.0055,0.1929,0.2231,0.2907,0.2259,0.3136,0.3302,0.3660,0.3956,0.4386,0.4670,0.5255,0.3735,0.2243,0.1973,0.4337,0.6532,0.5070,0.2796,0.4163,0.5950,0.5242,0.4178,0.3714,0.2375,0.0863,0.1437,0.2896,0.4577,0.3725,0.3372,0.3803,0.4181,0.3603,0.2711,0.1653,0.1951,0.2811,0.2246,0.1921,0.1500,0.0665,0.0193,0.0156,0.0362,0.0210,0.0154,0.0180,0.0013,0.0106,0.0127,0.0178,0.0231)
# changing inout data to numpy array
input_data_as_numpy_array = np.asarray(input_data)
# reshape the np array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = model.predict(input_data_reshaped)
print(prediction)