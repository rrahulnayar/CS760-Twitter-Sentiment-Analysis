
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import re
from sklearn import preprocessing

dataF = "G:/edu/sem2/760/proj/CS760-Twitter-Sentiment-Analysis/data/"
allF = dataF+'train_utf8.csv'
samF = dataF+'sample.csv'
featF = dataF+'svmAllFeatures.csv'


# In[2]:

dt = pd.read_csv(allF)
NUM_FEATURES = 7

negWords = re.compile(
    r'(not|no|nor|neither|(n\'*t)|false|wrong)\b'
,re.I)

posWords = re.compile(
    r'yes|true|correct'
,re.I)

wasteWords =re.compile(
    r'\b(the|a|an|as|be|am|is|are|was|were|has|have|had|'+
    r'at|from|to|in|under|before|after|near|far|away|for|and|'+
    r'that|them|there|their|they|his|her|hers|him|it|you|your|yours)\b'
, re.I)

#allow - &
badpunc = re.compile(
    r'\.|\_|\:|\'|\`|\"|\~|\!|\@|\#|\$|\%|\^|\*|\(|\)|\+|\=|\{|\}|\[|\]|'+
    r'\;|\<|\>|\,|\?|\/|\\|\|')
badwords = re.compile(r'\b(fuck|(f\*+))',re.I)

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
    #print(re.sub(wasteWords,'',txt).split())
    return len(re.sub(wasteWords,'',txt).split())

#count of neg words
def F4(txt):
    return len(re.findall(negWords,txt))

#count of pos words
def F5(txt):
    return len(re.findall(posWords,txt))
#contains bad words
def F6(txt):
    #print(re.findall(badwords,txt))
    return 1 if len(re.findall(badwords,txt))>0 else 0


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


# In[6]:

#correct all strings
tweets = dt['SentimentText'].map(modify)
tweets.to_csv('../data/tweetsModified.csv')
#Build all features
features = {i:tweets.map(f) for i,f in feats.items()}#ma


# In[7]:

#Normalize features
ft = pd.concat(features,1)
ft.to_csv('../data/features.csv')
#Make Combined database
normedFeats = pd.DataFrame(preprocessing.scale(ft))
dataFrame = pd.DataFrame(labels)
finalFeats = dataFrame.join(normedFeats)


# In[8]:

#save to file for SVM learning
finalFeats.to_csv(featF, index=False)


# In[9]:

#tw = "@cpuclub what the hell mike died???? Fuck man I always played for that guy in denver. Such a generous and kind dude. Rip "
#tw = modify(tw)
#print(tw,len(tw))
#features = {i:f(tw) for i,f in feats.items()}#ma
#print(features)

