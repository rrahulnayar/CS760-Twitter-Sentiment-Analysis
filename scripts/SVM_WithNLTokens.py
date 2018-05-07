
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import re
import sys
#import pp
#from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import TweetTokenizer


dataF = "G:/edu/sem2/760/proj/CS760-Twitter-Sentiment-Analysis/data/"
allF = dataF+'train_utf8.csv'
samF = dataF+'sample.csv'
featF = dataF+'svmVecFeatures.csv'

allF = sys.argv[1]
print(allF)
# In[2]:

dt = pd.read_csv(allF)
print('length: ',len(dt))

# In[3]:

#string corrections
def modify(txt):
    txt = txt.strip()
    txt = txt.replace('&quot;',"'")
    txt = txt.replace('&amp;','&')
    txt = txt.replace('&lt;','<')
    txt = txt.replace('&gt;','>')
    txt = re.sub(r'\s+',' ',txt)
    return txt


# In[4]:

def tokenize(text): 
    tknzr = TweetTokenizer()
    return tknzr.tokenize(text)

def stem(doc):
    return (stemmer.stem(w) for w in analyzer(doc))

en_stopwords = set(stopwords.words("english")) 

vectorizer = TfidfVectorizer(
    analyzer = 'word',
    tokenizer = tokenize,
    lowercase = True,
    ngram_range=(1, 1),
    stop_words = en_stopwords,
    max_features=512)


# In[5]:

vecs = vectorizer.fit_transform(dt['SentimentText'])
print(type(vecs))


# In[6]:

print(vecs.shape)


# In[7]:

feats = pd.DataFrame(vecs.toarray())
feats = pd.DataFrame(dt['Sentiment']).join(feats)
feats.to_csv(featF,index=False)
#type(vecs)


# In[8]:

# Split into train val and test sets 80-10-10
train,test = train_test_split(dt, test_size=0.1, shuffle=False)
#val,test = train_test_split(test, test_size=0.5, shuffle=False)
#Divide into X and Y
X_train,X_test = [t['SentimentText'] for t in [train,test]]
y_train,y_test = [t['Sentiment'] for t in [train,test]]
print([len(t) for t in [X_train,X_test,y_train,y_test]])


# In[ ]:

np.random.seed(1)
kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
pipeline_svm = make_pipeline(vectorizer, 
                            SVC(probability=True, kernel="linear", class_weight="balanced", verbose=True))

#grid_svm = GridSearchCV(pipeline_svm,
#                    param_grid = {'svc__C': [0.01, 0.1, 1]}, 
#                    cv = kfolds,
#                    scoring="roc_auc",
#                    verbose=1,   
#                    n_jobs=-1) 

#grid_svm.fit(X_train, y_train)
#grid_svm.score(X_test, y_test)


# In[ ]:

pipeline_svm.fit(X_train, y_train)


# In[ ]:

h_test_prob = pd.DataFrame(pipeline_svm.predict_proba(X_test), X_test.index)
h_test = h_test_prob[1]>0.5
h_test = h_test.map(lambda x:1 if x else 0)
h_test = h_test.rename('Predicted')
out_test = pd.concat([h_test_prob,h_test,y_test],axis=1)
out_test = out_test.rename(columns={'Sentiment':'Actual'})
out_test.to_csv('../data/Test_prob_embed_features_svm.csv',index=False)


# In[ ]:

def report_results(model, X, y):
    pred_proba = model.predict_proba(X)[:, 1]
    pred = model.predict(X)        

    auc = roc_auc_score(y, pred_proba)
    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred)
    prec = precision_score(y, pred)
    rec = recall_score(y, pred)
    result = {'auc': auc, 'f1': f1, 'acc': acc, 'precision': prec, 'recall': rec}
    return result



# In[ ]:

#report_results(grid_svm.best_estimator_, X_test, y_test)


# In[ ]:

TP = len(out_test[(out_test.Predicted==1) & (out_test.Actual==1)])
FP = len(out_test[(out_test.Predicted==1) & (out_test.Actual!=1)])
TN = len(out_test[(out_test.Predicted!=1) & (out_test.Actual!=1)])
FN = len(out_test[(out_test.Predicted!=1) & (out_test.Actual==1)])
print('counts: ',TP,TN,FP,FN)


# In[ ]:

Pr = TP/(TP+FP)
Rc = TP/(TP+FN)
F1 = 2*Pr*Rc/(Pr+Rc)
print('Score: ',Pr,Rc,F1)

