import pandas as pd
import sklearn.datasets
from sklearn.preprocessing import LabelEncoder

cancer_data = sklearn.datasets.load_breast_cancer()


df = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)


df['diagnosis'] = cancer_data.target


# print(df['diagnosis'].value_counts())

label_encode = LabelEncoder()

labels = label_encode.fit_transform(df.diagnosis)

df['target'] = labels

# print(df.head())

print(df['target'].value_counts())