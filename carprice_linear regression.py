# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 22:58:57 2020

@author: TITANS
"""

import pandas as pd
car_df  = pd.read_csv(r"C:\Users\TITANS\Downloads\crystal analytics\ml\Cars_Retail_Price .csv")

# exploring categorical variables:


for table in ['Cylinder', 'Doors', 'Cruise', 'Sound','Leather']:
    car_df[table] = car_df[table].astype('category')
    
    
   
cat_vars = car_df.select_dtypes(include='category').columns.tolist()
cat_vars=list(cat_vars)


for i in cat_vars:
    x=car_df[i].value_counts() 
    print(x)
    
    
    


# creating dummy variables(one hot encoding):

# create dummies for categorical variables:

car_df['Make'].value_counts()
dummies = pd.get_dummies(car_df['Make']).rename(columns=lambda x: 'Make_' + str(x))
# bring the dummies back into the original dataset
car_df = pd.concat([car_df, dummies], axis=1)

car_df['Model'].value_counts()
dummies2 = pd.get_dummies(car_df['Model']).rename(columns=lambda x: 'Model_' + str(x))
# bring the dummies back into the original dataset
car_df = pd.concat([car_df, dummies2], axis=1)

car_df['Trim'].value_counts()
dummies3 = pd.get_dummies(car_df['Trim']).rename(columns=lambda x: 'Trim_' + str(x))
# bring the dummies back into the original dataset
car_df = pd.concat([car_df, dummies3], axis=1)


car_df['Type'].value_counts()
dummies4 = pd.get_dummies(car_df['Type']).rename(columns=lambda x: 'Type_' + str(x))
# bring the dummies back into the original dataset
car_df = pd.concat([car_df, dummies4], axis=1)


 
car_df['Make'].value_counts()  # drop Make_Saturn
car_df['Model'].value_counts()  # keep Model_Malibu, Model_Cavalier, Model_AVEO, Model_Cobalt, Model_Ion
car_df['Trim'].value_counts()   # keep Sedan4D, Coupe 2D , LS Sedan 4D, LS Coupe 2D, LT Sedan 4D 
car_df['Type'].value_counts()   #drop convertible

 
new_df=car_df[['Price', 'Mileage','Cylinder','Liter','Doors','Cruise','Sound','Leather','Make_Buick','Make_Cadillac','Make_Chevrolet','Make_Pontiac','Make_SAAB','Model_Malibu','Model_Cavalier','Model_AVEO', 'Model_Cobalt',  'Model_Ion','Trim_Sedan 4D','Trim_Coupe 2D','Trim_LS Sedan 4D','Trim_LS Coupe 2D','Trim_LT Sedan 4D', 'Type_Coupe','Type_Hatchback','Type_Sedan','Type_Wagon']]


import numpy as np
x=np.var(new_df, axis=0)
x=pd.DataFrame(x)
x.to_csv("cars.csv")

import os
os.getcwd()

'''
#dropping low variannce variable
Trim_Coupe 2D
Trim_LS Sedan 4D
Trim_LS Coupe 2D
Trim_LT Sedan 4D
'''
new_df.drop(['Trim_Coupe 2D',
'Trim_LS Sedan 4D',
'Trim_LS Coupe 2D',
'Trim_LT Sedan 4D'],axis=1)
    
### dropping correlated var ####
 
cor=new_df.corr()
 
cor.to_csv('cor_linear_reg1.csv')   
import os
os.getcwd()
 
new_df_model=new_df.drop(['Trim_Coupe 2D','Type_Coupe'], axis=1)
 
 
 #converting all variables to numeric:
 
model_df = new_df_model.apply(pd.to_numeric)


model_df.drop(['Doors','Leather','Make_Chevrolet','Model_AVEO','Model_Ion','Trim_LS Sedan 4D'], axis=1,inplace=True)
model_df.drop(['Cylinder','Make_Buick','Make_Pontiac','Model_Cobalt','Trim_LT Sedan 4D'], axis=1,inplace=True)
model_df.drop(['Cruise'],axis=1,inplace=True) 



x = model_df.iloc[:,1:14]
y = model_df.iloc[:,0]
"""
##considering VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
#X=x_train_2.drop("const",axis=1)
vif['featues'] = x.columns
vif['VIF']=[variance_inflation_factor(x.values,i) for i in range(x.shape[1])]
vif['VIF'] = round(vif['VIF'],2)
vif=vif.sort_values(by='VIF',ascending=False)
vif


#dropping feature Liter since its VIF IS!<5
x=x.drop(["Liter"],axis=1)

"""
#Splitting the data into training and test sets
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state =64) 


#building model using statsmodel
import statsmodels.api as sm

x_train=sm.add_constant(x_train)
lm = sm.OLS(y_train,x_train).fit()
lm.summary()



#all stats principal seems to be satisfied
x_test=sm.add_constant(x_test)
y_pred = lm.predict(x_test)

stats=pd.concat([y_test,y_pred],axis=1) 
stats.rename(columns={"Price":"ACTUAL_PRICE"},inplace=True)
stats.rename(columns={0:"PREDICTED_PRICE"},inplace=True)
stats["Error_Price"]=stats["ACTUAL_PRICE"]-stats["PREDICTED_PRICE"]
stats['SQUARRED_ERROR'] = stats['Error_Price']**2
m_s_e=np.average(stats['SQUARRED_ERROR'])
r_m_s_e=np.sqrt(m_s_e)
avg_pred_price=np.average(stats["PREDICTED_PRICE"])
avg_actual_price=np.average(stats["ACTUAL_PRICE"])
error_percentage_price=(r_m_s_e/avg_pred_price)*100



import matplotlib.pyplot as plt
plt.xlabel("ACTUAL PRICE")
plt.ylabel("PREDICTED PRICE on TRAINING DATA")
plt.scatter(y_test,y_pred,  color='green')
plt.show()

from sklearn import metrics
import math as mt
mse = metrics.mean_squared_error(y_test, y_pred)
rmse=mt.sqrt(mse)




# model validtaion
#testing on shortlisted variables

var=list(x_train.columns)
x_test_final=x_test[var]
x_test_final=sm.add_constant()
y_pred_test=lm.predict(x_test_final)

from sklearn.metrics import r2_score
r2_test_score=r2_score(y_test,y_pred_test)

stats_t=pd.concat([y_test,y_pred_test],axis=1) 
stats_t.rename(columns={"Price":"ACTUAL_PRICE"},inplace=True)
stats_t.rename(columns={0:"PREDICTED_PRICE"},inplace=True)
stats_t["Error_Price"]=stats_t["ACTUAL_PRICE"]-stats_t["PREDICTED_PRICE"]
stats_t['SQUARRED_ERROR'] = stats_t['Error_Price']**2
m_s_e_t=np.average(stats_t['SQUARRED_ERROR'])
r_m_s_e_t=np.sqrt(m_s_e)
avg_pred_price_t=np.average(stats["PREDICTED_PRICE"])
avg_actual_price_t=np.average(stats["ACTUAL_PRICE"])
error_percentage_price_t=(r_m_s_e_t/avg_pred_price_t)*100


#VISUALISATION
import numpy as np
import seaborn as sns

#Scatter plot


import matplotlib.pyplot as plt
plt.scatter(y_test,y_pred_test,color='red')
plt.xlabel("ACTUAL PRICE")
plt.ylabel("PREDICTED PRICE on TEST DATA")

plt.show()
#distribution plot of error terms-

import seaborn as sb

plt.xlabel("errors_test",fontsize=15)
res = y_test - y_pred_test
plt.axvline(x=0,color="red")
sb.distplot(res,rug=True,color='blue')
import seaborn as sb

plt.xlabel("errors_training",fontsize=15)
res_train = y_test - y_pred
plt.axvline(x=0,color="red")
sb.distplot(res,rug=True,color='red')
