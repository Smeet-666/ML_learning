import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

diabetes_dataset = pd.read_csv('diabetes.csv')

# print(diabetes_dataset.head())
# print(diabetes_dataset.shape)  TO GET THE NO. OF ROWS AND COLUMNS

# TO GET THE STATISTICAL MEASURES
# print(diabetes_dataset.describe())

# print(diabetes_dataset['Outcome'].value_counts())


# 0 -->Non - diabetic
# 1 -->Diabetic

# print(diabetes_dataset.groupby('Outcome').mean())

X = diabetes_dataset.drop(columns = 'Outcome' , axis=1)
Y = diabetes_dataset['Outcome']
# print(X)
# print(Y)

scalar = StandardScaler()
scalar.fit(X)
standardized_data = scalar.transform(X)
# print(standardized_data)

X=standardized_data
Y = diabetes_dataset['Outcome']


# TRAIN TEST SPLIT

X_train , X_test ,Y_train , Y_test = train_test_split(X,Y, test_size=0.2 , stratify=Y , random_state=2)
# print(X.shape , X_train.shape , X_test.shape)

classifier = svm.SVC(kernel='linear')
classifier.fit(X_train,Y_train)

X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction , Y_test)
# print(test_data_accuracy)


# MAKING A PREDICTIVE SYSTEM

input_data = (1,97,66,15,140,23.2,0.487,22)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshape = input_data_as_numpy_array.reshape(1,-1)

# STANDARDIZING THE DATA
std_data = scalar.transform(input_data_reshape)
# print(std_data)

prediction = classifier.predict(std_data)
print(prediction)