
# coding: utf-8

# In[ ]:

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split

dataF = "G:/edu/sem2/760/proj/CS760-Twitter-Sentiment-Analysis/data/"
featF = dataF+'svmAllFeatures.csv'
NUM_FEATURES = 7


# In[ ]:

data = pd.read_csv(featF)#all data
data.columns


# In[ ]:

X = data.loc[:,'0':str(NUM_FEATURES-1)]
Y = data['Sentiment']
N = len(Y)#total rows
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, shuffle=False)
trN,vlN,tsN = len(y_train),len(y_val),len(y_test)


# In[ ]:

Y.unique()


# In[ ]:

model = SVC(verbose=True)#using rbf kernel, tol = 1e-3(default)
model.fit(X_train,y_train)


# In[ ]:

h_val = pd.Series(model.predict(X_val), X_val.index)#.reindex(X_val.index)
val = pd.concat([h_val,y_val],axis=1)
val = val.rename(columns={0:'Predicted', 'Sentiment':'Actual'})
val.to_csv('../data/Validation_set_predict_custom_features_svm.csv',index=False)
h_test = pd.Series(model.predict(X_val), X_test.index)#.reindex(X_val.index)
test = pd.concat([h_test,y_test],axis=1)
test = test.rename(columns={0:'Predicted', 'Sentiment':'Actual'})
test.to_csv('../data/Test_set_predict_custom_features_svm.csv',index=False)

# In[ ]:

TP = len(val[(val.Predicted=='Y') & (val.Actual=='Y')])
FP = len(val[(val.Predicted=='Y') & (val.Actual=='N')])
TN = len(val[(val.Predicted=='N') & (val.Actual=='N')])
FN = len(val[(val.Predicted=='N') & (val.Actual=='Y')])
TP,TN,FP,FN


# In[ ]:

Pr = TP/(TP+FP)
Rc = TP/(TP+FN)
F1 = 2*Pr*Rc/(Pr+Rc)
print(Pr,Rc,F1)
#0.6595561918396564 0.7877906976744186 0.7179926751344191

