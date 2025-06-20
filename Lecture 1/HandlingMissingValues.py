import numpy as np
import pandas as pd
import seaborn as sns

df = sns.load_dataset('titanic')
dataset = pd.DataFrame(df)

print(dataset.isnull().sum())   #before filling the missing values

# dataset['age'].fillna(dataset['age'].mean(), inplace=True)  # to fill the median values

# dataset = dataset.dropna(how='any')   # it will remove all the missing values from all the columns

print(dataset.isnull().sum())   #after filling the missing values

