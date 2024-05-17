#!/usr/bin/env python
# coding: utf-8

# In[6]:


#Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns
import datetime as dt
import time
from datetime import date, timedelta
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
get_ipython().run_line_magic('matplotlib', 'inline')

from google.oauth2.service_account import Credentials
import logging

from oauth2client.service_account import ServiceAccountCredentials

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

import sqlalchemy as db
from sqlalchemy import create_engine
import mysql.connector
import psycopg2

from powerbiclient import Report, models
from powerbiclient.authentication import MasterUserAuthentication
from powerbiclient.authentication import ServicePrincipalAuthentication
from powerbiclient.authentication import DeviceCodeLoginAuthentication
from powerbiclient.authentication import InteractiveLoginAuthentication
from io import StringIO
import ipywidgets
from ipywidgets import interact
import requests


# In[7]:


MyData = pd.read_csv('C:/Users/leemn/Downloads/Income_Expense_Data.csv')
MyData.head()


# In[10]:


MyData.isnull().sum()/len(MyData)*100


# In[11]:


MyData["Income"].fillna((MyData["Income"].median()), inplace = True)


# In[12]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.boxplot(MyData['Age'])
plt.show()


# In[13]:


Age_col_df = pd.DataFrame(MyData['Age'])
Age_median = Age_col_df.median()

#getting IQR of Age column
Q3 = Age_col_df.quantile(q=0.75)
Q1 = Age_col_df.quantile(q=0.25)
IQR = Q3-Q1

#Deriving boundaries of Outliers
IQR_LL = int(Q1 - 1.5*IQR)
IQR_UL = int(Q3 + 1.5*IQR)

#Finding and treating outliers - both lower and upper end
MyData.loc[MyData['Age']>IQR_UL , 'Age'] = int(Age_col_df.quantile(q=0.99))
MyData.loc[MyData['Age']<IQR_LL , 'Age'] = int(Age_col_df.quantile(q=0.01))


# In[14]:


x = MyData["Income"]
y=  MyData["Expense"]


plt.scatter(x, y, label="Income Expense")


# In[15]:


#check correltion matrix - to check the strength of variation bwtween two variables
correlation_matrix= MyData.corr().round(2)
f, ax = plt.subplots(figsize =(8, 4)) 
import seaborn as sns
sns.heatmap(data=correlation_matrix, annot=True)


# In[16]:


################feature engineering#######################
#Normalization/scaling of data - understanding scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(MyData)
scaled_data


# In[17]:


#converting data back to pandas dataframe
MyData_scaled = pd.DataFrame(scaled_data)
MyData_scaled.columns = ["Age","Income","Expense"]


# In[18]:


#Separating features and response
features = ["Income","Age"]
response = ["Expense"]
X=MyData_scaled[features]
y=MyData_scaled[response]


# In[19]:


#Importing neccesary packages
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split


# In[20]:


#Dividing data in test and train

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[22]:


#Fitting lineaar regression model
model = LinearRegression()
model.fit(X_train, y_train)
accuracy = model.score(X_test,y_test)
print(accuracy*100,'%')


# In[23]:


model.predict(X_test) 


# In[24]:


#Dumping the model object
import pickle
pickle.dump(model, open('model.pkl','wb'))


# In[26]:


#Reloading the model object
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[30000, 24]]))


# In[ ]:




