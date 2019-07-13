
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import sklearn.ensemble as sk
from sklearn.model_selection import train_test_split
from sklearn_pandas import DataFrameMapper, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import string
import nltk
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn import svm
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report


# In[2]:


df = pd.read_csv('train_file.csv')
df = df.fillna('')


# In[3]:


df.head()


# In[4]:


def text_process(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# In[5]:


x = df['Title']


# In[6]:


bow_transformer = TfidfVectorizer(analyzer=text_process).fit(x)
X = bow_transformer.transform(x)


# In[7]:


y = df['MaterialType']
le = LabelEncoder()
y=le.fit_transform(y)


# In[8]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[9]:


def fitpred(nb):
    print(nb)
    nb.fit(X_train, y_train)
    preds = nb.predict(X_test)
    mat = confusion_matrix(y_test, preds)
    print(mat)
    acc=(mat[0][0]+mat[1][1])/(mat[0][0]+mat[1][1]+mat[0][1]+mat[1][0])
    print(acc)
    print('\n')
    print(classification_report(y_test, preds))


# In[11]:


nb = MultinomialNB()
fitpred(nb)
nb2 = BernoulliNB()
fitpred(nb2)
clf = svm.SVC(gamma='scale')
fitpred(clf)
clf2 = xgb.XGBClassifier()
fitpred(clf2)
clf3 = DecisionTreeClassifier()
fitpred(clf3)
clf4 = RandomForestClassifier()
fitpred(clf4)


# In[12]:


nbf = xgb.XGBClassifier()
nbf.fit(X, y)


# In[13]:


test = pd.read_csv('test_file.csv')
test.fillna('')
X_t = test['Title']
X_t = bow_transformer.transform(X_t)
preds = nbf.predict(X_t)
preds = le.inverse_transform(preds)


# In[14]:


id_ = test['ID']
label = preds
data = { 'ID': id_, 'MaterialType': label}
submission = pd.DataFrame(data)
submission.head(10)


# In[15]:


submission.to_csv('submit.csv',index=False)

