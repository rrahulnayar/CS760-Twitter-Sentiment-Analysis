
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import re
from sklearn import preprocessing

dataF = "../data/"
allF = dataF+'train_utf8.csv'
samF = dataF+'sample.csv'
featF = dataF+'svmAllFeatures.csv'


# In[2]:

dt = pd.read_csv(samF)
NUM_FEATURES = 6

negWords = re.compile(
    r'(not|no|nor|neither|(n\'*t)|false|wrong)\b'
,re.I)

posWords = re.compile(
    r'yes|true|correct'
,re.I)

wasteWords =re.compile(
    r'\b(the|a|an|as|be|am|is|are|was|were|has|have|had|\
    at|from|to|in|under|before|after|near|far|away|\
    that|them|there|their|they|his|her|hers|him|it|you|your|yours|)\b'
, re.I)

#allow - &
badpunc = re.compile(
    r'\.|\_|\:|\'|\`|\"|\~|\!|\@|\#|\$|\%|\^|\*|\(|\)|\+|\=|\{|\}|\[|\]|\
    \;|\<|\>|\,|\?|\/|\\|\|')

#tweet character length
def F0(txt):
    return len(txt)

#tweet word length
def F1(txt):
    return len(txt.split())

#tweet character length w/o punc
def F2(txt):
    return len(re.sub(badpunc,'',txt))

#word length w/o atricles and prepositions
def F3(txt):
    return len(re.sub(wasteWords,'',txt).split())

#count of neg words
def F4(txt):
    return len(re.findall(negWords,txt))

#count of pos words
def F5(txt):
    return len(re.findall(posWords,txt))


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

labels = dt['Sentiment'].map(lambda x: 'Y' if x==1 else 'N')


# In[5]:

#Loop of features
feats = {str(i):eval('F'+str(i)) for i in range(NUM_FEATURES)}
#correct all strings
tweets = dt['SentimentText'].map(modify)
#Build all features
features = {i:tweets.map(f) for i,f in feats.items()}#ma


# In[6]:

#Normalize features
ft = pd.concat(features,1)
normedFeats = pd.DataFrame(preprocessing.scale(ft))


# In[7]:

#Make complete data frame by appending label column, do this only once
finalFeats = normedFeats.join(labels)
#save to file for SVM learning
finalFeats.to_csv(featF, index=False)

