
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from sklearn.svm import SVC
import sys

print(sys.argv)
dataF = '../data/'
vecsF = sys.argv[1]#dataF+'allVecs.csv'
suffix = sys.argv[3]
outF = dataF+'SVM_Output_'+suffix+'.csv'
probsF = dataF+'SVM_Confidence_'+suffix+'.txt'


# In[2]:

dt = pd.read_csv(vecsF)
print(len(dt), dt.columns,dt.Label.unique())


# In[3]:

NumFeatures = len(dt.columns)-1
N = len(dt)


# In[4]:

train,test = dt[:(N-10000)],dt[(N-10000):]
X_train,X_test = [t.loc[:,'0':str(NumFeatures-1)] for t in [train,test]]
Y_train,Y_test = [t.Label for t in [train,test]]
len(X_train),len(X_test),len(Y_train),len(Y_test)


# In[5]:

N_train = int(sys.argv[2])#1000
X_train2,Y_train2 = X_train[:N_train],Y_train[:N_train]


# In[6]:

model = SVC(verbose=True, probability=True)


# In[7]:

model.fit(X_train2,Y_train2)


# In[8]:

h_test_prob = pd.DataFrame(model.predict_proba(X_test), X_test.index)


# In[9]:

h_test = h_test_prob[1]>0.5#Threshold = 50%
h_test = h_test.map(lambda x:'Y' if x else 'N')
h_test = h_test.rename('Predicted')
out_test = pd.concat([h_test_prob,h_test,Y_test],axis=1)
#out_test = out_test.rename(columns={'Sentiment':'Actual'})
out_test.to_csv(outF,index=False)
h_test_prob[1].to_csv(probsF,index=False)


# In[11]:

TP = len(out_test[(out_test.Predicted=='Y') & (out_test.Label=='Y')])
FP = len(out_test[(out_test.Predicted=='Y') & (out_test.Label!='Y')])
TN = len(out_test[(out_test.Predicted!='Y') & (out_test.Label!='Y')])
FN = len(out_test[(out_test.Predicted!='Y') & (out_test.Label=='Y')])
out_counts = {'TP':TP,'FP':FP,'TN':TN,'FN':FN}


# In[12]:

Pr = TP/(TP+FP)
Rc = TP/(TP+FN)
F1 = 2*Pr*Rc/(Pr+Rc)
out_scores= {'Pr':Pr,'Rc':Rc,'F1':F1}
with open(dataF+suffix+'.txt','w') as fd:
    fd.write(str({**out_counts,**out_scores}))

