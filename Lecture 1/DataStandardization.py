import numpy as np
import pandas as pd
import  sklearn.datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

dataset = sklearn.datasets.load_breast_cancer()

# print(dataset)

df =pd.DataFrame(dataset.data , columns=dataset.feature_names)

# print(df.head())
# print(df.shape)

X= df
Y = dataset.target

# print(X)
# print(Y)

X_train , X_test , Y_train , Y_test = train_test_split(X,Y , train_size=0.2 , random_state=3 )

# print(X.shape , X_train.shape , X_test.shape)

# print(dataset.data.std())

scalar = StandardScaler()
scalar.fit(X_train)
X_train_standardized  = scalar.transform(X_train)

# print(X_train_standardized)

X_test_standardized = scalar.transform(X_test)

print(X_train_standardized.std())
print(X_test_standardized.std())
