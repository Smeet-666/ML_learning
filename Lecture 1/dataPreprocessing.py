import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

diabetes_data = pd.read_csv('diabetes.csv')
# print(diabetes_data.head())

# print(diabetes_data.shape)
# print(diabetes_data.describe())

# separating feeatures and targets
X = diabetes_data.drop(columns='Outcome' , axis=0)
Y = diabetes_data['Outcome']
# print(X)
# print(Y)

# 0 --> diabetic patient
# 1 --> Non - diabetic patient

# Data standardization

scalar = StandardScaler()
standardized_data = scalar.fit_transform(X)
# print(standardized_data)

X = standardized_data
X_train , X_test , Y_train , Y_test = train_test_split(X,Y , test_size=0.2 , random_state=2)
print(X.shape , X_test.shape , X_train.shape)