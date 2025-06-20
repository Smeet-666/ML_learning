# Linear Regression

import numpy as np
import pandas as pd
import  seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv("height_weight_dataset.csv")
# print(df.head())
# plt.scatter(df['Weight_kg'] , df['Height_cm'])
# plt.xlabel("Widht")
# plt.ylabel("Height")

# sns.pairplot(df)
# plt.show()

X=[['Weight_kg']]
y=['Height_cm']


from sklearn.model_selection import train_test_split
X_train , X_test ,y_train , y_test = train_test_split(X,y,train_size=0.25 ,random_state=42)
print(X_train.shape)