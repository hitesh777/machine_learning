# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 12:15:46 2020

@author: TITANS
"""

### Objective: Cancer type detection algorithm- classification model.


from sklearn.tree     import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd


cancer_data = pd.read_csv(r"C:\Users\TITANS\Downloads\cancer_data.csv")

#checking data types:
cancer_data.dtypes

#checking missing values:
cancer_data.info()   #no missing values were found

#checking event rate
cancer_data['diagnosis'].value_counts()

# understanding other features:
cancer_data.describe()

#droping useless variables:
cancer_data=cancer_data.drop(['id','Unnamed: 32'],axis=1)




df=cancer_data 
df.shape
df['diagnosis'].value_counts()

# balancing class data set

from sklearn.utils import resample



# Separate majority and minority classes
df_majority = df[df.diagnosis=='B']
df_minority = df[df.diagnosis=='M']

# Upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=357,    # to match majority class
                                 random_state=123) # reproducible results



# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
df=df_upsampled
df.diagnosis.value_counts()






# Create correlation matrix
import numpy  as np
corr_matrix = cancer_data.corr().abs()
# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.60
to_drop_up = [column for column in upper.columns if any(upper[column] > 0.60)]



# Drop features 
df.drop(df[to_drop_up], axis=1,inplace=True)



#data prep for model:

#shuffling
from sklearn import utils as ut
#shuffiling dataset
df=ut.shuffle(df)

#The feature columns into X, and the label column ‘Diagnosis’ into Y:

X=df.iloc[:,1:]
Y=df.iloc[:,0]  #dependent var


#changing dependent variable into 0/1

Y=Y.map({'B':0,'M':1})





#modeling



# split X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=12)



#Hyperparameter Optimization: Gridsearch CV
'''
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from urllib.request import urlopen
from sklearn.model_selection import cross_val_score
np.random.seed(42)
start = time.time()

param_dist = {'max_depth': [2, 3, 4],
              'n_estimators':[100,200,300,400],
              'max_features': ['auto', 'sqrt', 'log2', None],
              'criterion': ['gini', 'entropy']}

cv_rf = GridSearchCV(fit_rf, cv = 10,
                     param_grid=param_dist,
                     n_jobs = 3)
     
cv_rf.fit(X_train, y_train)
print('Best Parameters using grid search: \n',
      cv_rf.best_params_)
end = time.time()
print('Time taken in grid search: {0: .2f}'.format(end - start))

#Run random forest on hyperparameters choosen
'''
RF=RandomForestClassifier(bootstrap=True,  oob_score = False, class_weight=None, criterion='entropy',
            max_depth=4, max_leaf_nodes=None, max_features= 'auto',
            n_estimators=300, n_jobs=3, random_state=42,
            verbose=0, warm_start=True)

RF.fit(X_train, y_train)

# make predictions for train data
y_pred = RF.predict_proba(X_train)
prob_2 = pd.DataFrame(y_pred)
prob_pred=prob_2[1]
y_pred_t = prob_pred.apply(lambda x: 1 if x > 0.50 else 0)





# evaluate model accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_train, y_pred_t)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


#confusion matrix
import pandas as pd
CF=confusion_matrix(y_train, y_pred_t)
print(CF)

import matplotlib.pyplot as plt
import seaborn as sns
# Get and reshape confusion matrix data
matrix = confusion_matrix(y_train, y_pred_t)
matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

# Build the plot
plt.figure(figsize=(16,7))
sns.set(font_scale=1.4)
sns.heatmap(matrix, annot=True, annot_kws={'size':10},
            cmap=plt.cm.Greens, linewidths=0.2)

# Add labels to the plot
class_names = ['NO DISEASE','CANCER']
tick_marks = np.arange(len(class_names))
tick_marks2 = tick_marks + 0.4
plt.title("CONFUSION MATRIX -TRAIN DATA")
plt.xticks(tick_marks, class_names, rotation=0)
plt.yticks(tick_marks2, class_names, rotation=0)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix for Random Forest Model')
plt.show()
y_train.value_counts()


#classification_report
CR=classification_report(y_train, y_pred_t)
print(CR)






#ROC
from sklearn.metrics import roc_curve, auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train,y_pred_t)
roc_auc = auc(false_positive_rate, true_positive_rate)
print("ROC AUC ", roc_auc)



from sklearn.metrics import roc_curve
# compute true positive rate and false positive rate
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train,y_pred_t)
# plotting them against each other
def plot_roc_curve(false_positive_rate, true_positive_rate, label=None):
    plt.plot(false_positive_rate, true_positive_rate, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'r', linewidth=4)
    plt.axis([0, 1, 0, 1])
    plt.title("ROC-TRAINING CURVE")
    plt.xlabel('False Positive Rate (FPR)', fontsize=16)
    plt.ylabel('True Positive Rate (TPR)', fontsize=16)

plt.figure(figsize=(14, 7))
plot_roc_curve(false_positive_rate, true_positive_rate)
plt.show()



# SAVING TRAINED MODEL
# save the model to disk
import pickle
filename = 'RF_model_96.sav'
pickle.dump(RF, open(filename, 'wb'))
 
# some time later...
 
# load the model from disk
'''
rf_loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(x_test, y_test)
print(result)
'''


# make predictions for test data
y_pred = rf_loaded_model.predict_proba(X_test)
prob_2 = pd.DataFrame(y_pred)
prob_pred=prob_2[1]
y_pred_new = prob_pred.apply(lambda x: 1 if x > 0.50 else 0)



# evaluate model accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred_new)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


#confusion matrix
import pandas as pd
CF=confusion_matrix(y_test, y_pred_new)
print(CF)

import seaborn as sns
# Get and reshape confusion matrix data
matrix = confusion_matrix(y_test, y_pred_new)
matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

# Build the plot
plt.figure(figsize=(16,7))
sns.set(font_scale=1.4)
sns.heatmap(matrix, annot=True, annot_kws={'size':10},
            cmap=plt.cm.Greens, linewidths=0.2)

# Add labels to the plot
class_names = ['NO DISEASE','CANCER']
tick_marks = np.arange(len(class_names))
tick_marks2 = tick_marks +0.5
plt.xticks(tick_marks, class_names, rotation=0)
plt.yticks(tick_marks2, class_names, rotation=0)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix for Random Forest Model')
plt.show()
y_train.value_counts()


#classification_report
CR=classification_report(y_test, y_pred_new)
print(CR)


#ROC
from sklearn.metrics import roc_curve, auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test,y_pred_new)
roc_auc = auc(false_positive_rate, true_positive_rate)
print("ROC AUC ", roc_auc)



from sklearn.metrics import roc_curve
# compute true positive rate and false positive rate
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test,y_pred_new)
# plotting them against each other
def plot_roc_curve(false_positive_rate, true_positive_rate, label=None):
    plt.plot(false_positive_rate, true_positive_rate, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'r', linewidth=4)
    plt.title("ROC-TESTING CURVE")
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate (FPR)', fontsize=16)
    plt.ylabel('True Positive Rate (TPR)', fontsize=16)

plt.figure(figsize=(14, 7))
plot_roc_curve(false_positive_rate, true_positive_rate)
plt.show()
        
        
        

# variable importance:

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
# Get our features and weights
feature_list = sorted(zip(map(lambda x: round(x, 3), RF.feature_importances_), X_train.columns),
             reverse=True)
# Print them out
print('feature\t\timportance')
print("\n".join(['{}\t\t{}'.format(f,i) for i,f in feature_list]))
print('total_importance\t\t',  sum([i for i,f in feature_list]))