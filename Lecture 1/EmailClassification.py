import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

raw_mail_data = pd.read_csv('emails.csv')
# print(raw_mail_data.head())
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)) , ' ')

# print(mail_data.head())

#checking rows and columns
print(mail_data.shape) 