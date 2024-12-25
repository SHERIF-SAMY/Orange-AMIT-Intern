
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import re 
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def preprocessing_step(text):
  text = text.lower()
  ### Remove any special charchter
  text =re.sub('[^a-zA-Z]',' ',text)
  tokens = word_tokenize(text)
  stop_words = set(stopwords.words('english'))
  filtered_tokens = [word for word in tokens if word not in stop_words]
  stemmer = PorterStemmer()
  stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
  stemmed_tokens = ' '.join(stemmed_tokens)
  return stemmed_tokens

# # Preprocessing function
# def preprocessing_step(text):
#     # Tokenization
#     tokens = word_tokenize(text.lower())

#     # Stopword removal
#     stop_words = set(stopwords.words('english'))
#     filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]

#     # Lemmatization
#     lemmatizer = WordNetLemmatizer()
#     lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

#     # Join tokens back into a string
#     return ' '.join(lemmatized_tokens)
