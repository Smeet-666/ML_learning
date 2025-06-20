import numpy as np
import pandas as pd
import seaborn as sns

credit_data = pd.read_csv('creditcard.csv')
# print(credit_data.head())

# separating the legit and fraudulent data
legit = credit_data[credit_data.Class == 0]
fraud = credit_data[credit_data.Class == 1]

# print(legit.shape)   # legit--> 284315
# print(fraud.shape)   #fraud--> 492


legit_sample = legit.sample(n = 492)
# print(legit_sample.shape)       #now the no of legit sample are also 492


#concating the new dataset
new_dataset=pd.concat([legit_sample , fraud] , axis=0 )
print(new_dataset.head())
