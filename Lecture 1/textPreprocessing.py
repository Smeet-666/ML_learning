import pandas as pd
import numpy as np
import  re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# nltk.download('stopwords')
# print(stopwords.words('english'))

# DATA PREPROCESSING

news_data = pd.read_csv('fake.csv')
# print(news_data.head() )

# print(news_data.shape)

# CHECKING FOR MISSING VALUES
# print(news_data.isnull().sum())

# IF THERE IS ANY MISSING VALUE IN STRING WE CAN REPLACE THEM WITH A NULL STRING
# news_data = news_data.fillna('')


# STEMMING
port_stem = PorterStemmer()
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)     # remove non-alphabets
    stemmed_content = stemmed_content.lower()               # convert to lowercase
    stemmed_content = content.split()                               # tokenize
    stemmed_content = [port_stem.stem(word) for word in content if
               word not in stopwords.words('english')]  # stopword removal + stemming
    return ' '.join(content)
    return stemmed_content

news_data['content'] = news_data['text'].apply(stemming)

X = news_data['content'].values
Y = news_data['label'].values

print(Y )