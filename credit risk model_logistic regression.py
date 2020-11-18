# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 23:20:13 2020

@author: TITANS
"""

# linear algebra
import numpy as np 

# data processing
import pandas as pd 

# data visualization
from matplotlib import pyplot as plt
from matplotlib import style

# Algorithms
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression



# reading data:

import pandas as pd
Good_bad = pd.read_csv(r"C:\Users\TITANS\Downloads\good_bad.csv")
Good_bad.shape

#Data Exploration/Analysis


#checking missing values:
Good_bad.info()   #no missing values were found

#Let's take a more detailed look at what data is actually missing:
 
total = Good_bad.isnull().sum().sort_values(ascending=False)       #total numbers of missing values in data sorted by highest to lowest order
percent_1 = Good_bad.isnull().sum()/Good_bad.count()*100
percent_2 = (round(percent_1, 1)).sort_values(ascending=False)    # order of desc percentage of missing rate(round off to 1 decimal)
missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])

# outliers:

x=pd.DataFrame(Good_bad.describe())

#x['Duration', 'Amount', 'Rate', 'CurrResidTenure', 'Age', 'ExCredit', 'NumLiab']

new = Good_bad[['Duration', 'Amount', 'Rate', 'CurrResidTenure', 'Age', 'ExCredit', 'NumLiab']].copy()


def outlier_detect(df):
    for i in df.describe().columns:
        Q1=df.describe().at['25%',i]
        Q3=df.describe().at['75%',i]
        IQR=Q3 - Q1
        LTV=Q1 - 1.5 * IQR
        UTV=Q3 + 1.5 * IQR
        x=np.array(df[i])
        p=[]
        for j in x:
            if j < LTV or j>UTV:
                p.append(df[i].median())
            else:
                p.append(j)
        df[i]=p
    return df



xf=outlier_detect(new)
treated=pd.DataFrame(xf.describe())

s1=Good_bad.columns.tolist()
s2=xf.columns.tolist()
s3=set(s1)-set(s2)
print(s3)
n2=Good_bad[['EmployTenure', 'Plans', 'Foreign', 'CreditHistory', 'Check_Account_Status', 'Purpose', 'Tel', 'Debtors', 'Good/Bad', 'Job', 'Status', 'SavingsAcc', 'Propert', 'Hous']].copy()



Good_bad=pd.concat([n2,xf],axis=1)


Good_bad.describe()


Good_bad["Amount"]=Good_bad["Amount"].apply(np.sqrt)
Good_bad.shape


# create dummies for categorical variables:
import pandas as pd
Good_bad['CreditHistory'].value_counts()

dummies = pd.get_dummies(Good_bad['CreditHistory']).rename(columns=lambda x: 'CreditHistory_' + str(x))
# bring the dummies back into the original dataset
Good_bad = pd.concat([Good_bad, dummies], axis=1)

Good_bad['Purpose'].value_counts()
dummies2 = pd.get_dummies(Good_bad['Purpose']).rename(columns=lambda x: 'Purpose_' + str(x))
# bring the dummies back into the original dataset
Good_bad = pd.concat([Good_bad, dummies2], axis=1)



dummies3 = pd.get_dummies(Good_bad['Check_Account_Status']).rename(columns=lambda x: 'Check_Account_Status_' + str(x))
# bring the dummies back into the original dataset
Good_bad = pd.concat([Good_bad, dummies3], axis=1)


dummies4 = pd.get_dummies(Good_bad['SavingsAcc']).rename(columns=lambda x: 'SavingsAcc_' + str(x))
# bring the dummies back into the original dataset
Good_bad = pd.concat([Good_bad, dummies4], axis=1)


dummies5 = pd.get_dummies(Good_bad['EmployTenure']).rename(columns=lambda x: 'EmployTenure_' + str(x))
# bring the dummies back into the original dataset
Good_bad = pd.concat([Good_bad, dummies5], axis=1)



dummies6 = pd.get_dummies(Good_bad['Status']).rename(columns=lambda x: 'Status_' + str(x))
# bring the dummies back into the original dataset
Good_bad = pd.concat([Good_bad, dummies6], axis=1)



dummies7 = pd.get_dummies(Good_bad['Debtors']).rename(columns=lambda x: 'Debtors_' + str(x))
# bring the dummies back into the original dataset
Good_bad = pd.concat([Good_bad, dummies7], axis=1)



dummies8 = pd.get_dummies(Good_bad['Propert']).rename(columns=lambda x: 'Propert_' + str(x))
# bring the dummies back into the original dataset
Good_bad = pd.concat([Good_bad, dummies8], axis=1)



dummies9 = pd.get_dummies(Good_bad['Plans']).rename(columns=lambda x: 'Plans_' + str(x))
# bring the dummies back into the original dataset
Good_bad = pd.concat([Good_bad, dummies9], axis=1)



dummies10 = pd.get_dummies(Good_bad['Hous']).rename(columns=lambda x: 'Hous_' + str(x))
# bring the dummies back into the original dataset
Good_bad = pd.concat([Good_bad, dummies10], axis=1)




dummies11 = pd.get_dummies(Good_bad['Job']).rename(columns=lambda x: 'Job_' + str(x))
# bring the dummies back into the original dataset
Good_bad = pd.concat([Good_bad, dummies11], axis=1)



dummies12 = pd.get_dummies(Good_bad['Tel']).rename(columns=lambda x: 'Tel_' + str(x))
# bring the dummies back into the original dataset
Good_bad = pd.concat([Good_bad, dummies12], axis=1)



dummies13 = pd.get_dummies(Good_bad['Foreign']).rename(columns=lambda x: 'Foreign_' + str(x))
# bring the dummies back into the original dataset
Good_bad = pd.concat([Good_bad, dummies13], axis=1)


Good_bad.rename(columns={'Good/Bad': 'Good_Bad'}, inplace=True)

Good_bad.dtypes


# drop columns that are not required- categorical + dummies(dummy variable trap ones+ lower frequency ones)

#---> drop original categorical var:
    
Good_bad.drop(['Check_Account_Status','CreditHistory','Purpose','SavingsAcc','EmployTenure','Status','Debtors','Propert','Plans','Hous','Job','Tel','Foreign'],axis=1,inplace=True)
 
 
 #checking frequencies and dropping variables:
 
Good_bad.apply(lambda x: x.value_counts()).T.stack()
 
 
 
Good_bad['Foreign_A202'].value_counts()  #drop
Good_bad['Plans_A142'].value_counts()   #drop
Good_bad['Job_A171'].value_counts() #drop 
Good_bad['Tel_A192'].value_counts() #drop
Good_bad['Hous_A153'].value_counts() #drop 
Good_bad['Job_A171'].value_counts() #drop 
Good_bad['Propert_A124'].value_counts() #drop
Good_bad['Status_A91'].value_counts() #drop
Good_bad['Debtors_A102'].value_counts() #drop
Good_bad['EmployTenure_A71'].value_counts() #drop
Good_bad['SavingsAcc_A64'].value_counts() #drop
Good_bad['Check_Account_Status_A13'].value_counts() #drop
Good_bad['CreditHistory_A30'].value_counts() #drop
Good_bad['Purpose_A48'].value_counts() #drop
 
 
#-------->n-1 category drop:
 
Good_bad.drop(['Purpose_A44','Propert_A124','Status_A91','Debtors_A102','EmployTenure_A71','SavingsAcc_A64','Check_Account_Status_A13','CreditHistory_A30'],axis=1,inplace=True)

    
Good_bad.drop(['Purpose_A41'	,
'SavingsAcc_A62'	,
'Purpose_A49'	,
'Debtors_A101'	,
'Status_A94'	,
'CreditHistory_A33'	,
'SavingsAcc_A63'	,
'Debtors_A103'	,
'Purpose_A46'	,
'CreditHistory_A31'	,
'Plans_A142'	,
'Foreign_A201'	,
'Foreign_A202'	,
'Purpose_A45'	,
'Job_A171'	,
'Purpose_A410'	,
'Purpose_A48'	],axis=1,inplace=True)



Good_bad.drop(["CreditHistory_A34","CreditHistory_A32","SavingsAcc_A65","Status_A92","Plans_A141","Hous_A151","Hous_A153","Job_A172","Job_A174",'Tel_A191'],axis=1,inplace=True)


Good_bad['Good_Bad'] = Good_bad['Good_Bad'].apply(lambda x: 1 if x == 2 else 0)


# balancing class data set

from sklearn.utils import resample



# Separate majority and minority classes
df_majority = Good_bad[Good_bad.Good_Bad==0]
df_minority = Good_bad[Good_bad.Good_Bad==1]

# Upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=700,    # to match majority class
                                 random_state=123) # reproducible results



# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
Good_bad=df_upsampled
Good_bad.Good_Bad.value_counts()


Good_bad.apply(lambda x: x.value_counts()).T.stack()

#shuffling
from sklearn import utils as ut
#shuffiling dataset
Good_bad=ut.shuffle(Good_bad)


Good_Bad2=Good_bad[:]#making copy of preprocessed data
Good_Bad2.shape


# data for all independent var:
Good_Bad2.drop(['Good_Bad'],axis=1,inplace=True)
   

#split dataset in features and target variable

X = Good_Bad2 # IV's
y = Good_bad['Good_Bad'] # Target variable-DV


#model building
''''
independent var-26
dependent var-01


'''
# split X and y into training and testing sets
from sklearn.model_selection import train_test_split


from sklearn.linear_model import LogisticRegression

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=10)


# TRANSFORMATION
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range = (0,1))

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


logreg = LogisticRegression()
# fit the model with data
logreg.fit(x_train,y_train)

#finding probabilies of being a defaulter
probs = logreg.predict_proba(x_train)  

probs=pd.DataFrame(probs)

prob1=probs.iloc[:,1]# taking probability for Defaulter

probs=pd.DataFrame(prob1)

#Keep Probabilities of the positive class only.
probs.columns = ['p_d']

y_pred = probs['p_d'].apply(lambda x: 1 if x > 0.44 else 0)

y_pred.value_counts()

# actual vs pred:
#-- actual:
y_train.value_counts()

'''
0    479
1    501
'''
#-- pred:
y_pred.value_counts()
'''

0    463
1    517
'''



# confusion matrix- training

'''
from pandas_ml import ConfusionMatrix
ConfusionMatrix(y_train, y_pred)
'''

from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_train, y_pred)
cnf_matrix


#ROC curve- Compute the AUC Score.
from sklearn import metrics
auc = metrics.roc_auc_score(y_train, probs)  
print('AUC: %.2f' % auc)  
  
#Get the ROC Curve.
fpr, tpr, thresholds = metrics.roc_curve(y_train, probs)  

#Plot ROC Curve using our defined function
plt.title("Training -ROC")
plt.xlabel("False Positive Rate/SPECIFICITY/")
plt.ylabel("True Positive Rate/RECALL/SENSITIVITY")
plt.plot(fpr, tpr) 





# precision accuracy recall

accu=(metrics.accuracy_score(y_train, y_pred))

precision=metrics.precision_score(y_train, y_pred)*100
recall=metrics.recall_score(y_train, y_pred)*100
print("Accuracy:",metrics.accuracy_score(y_train, y_pred))
print("Precision:",metrics.precision_score(y_train, y_pred))
print("Recall:",metrics.recall_score(y_train, y_pred))

#F1 score

data=[precision , recall]
import statistics as st
F1_cal=st.harmonic_mean(data)*100
print("F1_SCORE calculated", F1_cal)

# SAVING TRAINED MODEL
# save the model to disk
import pickle
filename = 'LOGREG_finalized_model.sav'
pickle.dump(logreg, open(filename, 'wb'))
 
# some time later...
 
# load the model from disk
'''
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(x_test, y_test)
print(result)
'''
probs = loaded_model.predict_proba(x_test) #TESTING DATA
#probs = logreg.predict_proba(x_test)  

probs=pd.DataFrame(probs)

prob1=probs.iloc[:,1]

probs=pd.DataFrame(prob1)

#Keep Probabilities of the positive class only.
probs.columns = ['p_d']

y_pred = probs['p_d'].apply(lambda x: 1 if x > 0.44 else 0)
y_pred.value_counts()

#-- actual:
y_test.value_counts()

'''
1    199
0    221
'''
#-- pred:
y_pred.value_counts()
'''
1    231
0    189

'''


# confusion matrix- training
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix




#ROC curve- Compute the AUC Score.
from sklearn import metrics
auc = metrics.roc_auc_score(y_test, y_pred)  
print('AUC: %.2f' % auc)  
  
#Get the ROC Curve.
fpr, tpr, thresholds = metrics.roc_curve(y_test, probs)  

plt.title("Testing -ROC")
plt.xlabel("False Positive Rate/SPECIFICITY/")
plt.ylabel("True Positive Rate/RECALL/SENSITIVITY")
plt.plot(fpr, tpr)  





# precision accuracy recall

accu=metrics.accuracy_score(y_test, y_pred)
precision=metrics.precision_score(y_test, y_pred)
recall=metrics.recall_score(y_test, y_pred)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))


data=[precision , recall]
import statistics as st
F1_cal=st.harmonic_mean(data)
print("F1_SCORE calculated", F1_cal)

#F1 score
F1_score=metrics.f1_score(y_test, y_pred)

# Gini Score--

(2*auc)-1
1.52-1= .52...52%