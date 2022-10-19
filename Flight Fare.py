#!/usr/bin/env python
# coding: utf-8

# # Flight Fare EDA
# Have you ever struggled to book flight tickets at the cheapest rate? one day they are priced a certian amount and only in a few hours, the rate increases or decreases. Turns out their fare is dependent upon multiple factors. So, my attempt through this project is to identify and understand those fatcors influencing the prices and develop a model predicting flight prices 
# 
# For doing the analysis, I am using Flight Fare dataset from Kaggle. Link: https://www.kaggle.com/code/singhakash/flight-price-prediction/data 

# # Dataset Information
# * Airline: List airlines names like Indigo, Jet Airways, Air India, and many more.
# * Date_of_Journey:the date on which the passenger’s journey begins.
# * Source: the place from where the passenger’s journey will start.
# * Destination:where passengers wanted to travel.
# * Route: route opted by flights to reach from source to destination.
# * Arrival_Time: Arrival time is when the passenger will reach his/her destination.
# * Duration: Time takn by flights to complete the journey.
# * Total_Stops: stops taken by flight to reach from source to destination.
# * Additional_Info: information about food, kind of food, and other amenities.
# * Price: Price of the flight for a complete journey including all the expenses before onboarding.

# In[1]:


# importing all necessary libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns #for visualization

import matplotlib.pyplot as plt #for visualization


# In[2]:


# Importing the training dataset

flight_df=pd.read_excel('Data_Train.xlsx')


# In[3]:


flight_df.head()


# In[4]:


flight_df.info()


# So, we have 10 object datatype columns and only one int. In order to use these columns for predicition purposes, we will need to encode them. But first, let's see if we have any null values

# In[5]:


flight_df['Price'].describe()


# In[6]:


flight_df.isnull().sum()


# Since, we have only one null value in destination and total_stops, we'll just discard them

# In[7]:


flight_df.dropna(inplace=True)


# In[8]:


flight_df.isnull().sum()


# In[9]:


flight_df[flight_df.duplicated()]


# * there are 220 duplicate rows. Before proceeding forward, we'll remove them. Also, Delhi and New Delhi are same, so I'll be renaming Delhi to New Delhi and check for duplicates again.

# In[10]:


flight_df['Source'] = flight_df['Source'].replace({'Delhi': 'New Delhi'})
flight_df['Destination'] = flight_df['Destination'].replace({'Delhi': 'New Delhi'})


# In[11]:


flight_df[flight_df.duplicated()]


# In[12]:


# Removing all duplicates
flight_df.drop_duplicates(keep=False, inplace=True)


# In[13]:


flight_df[flight_df.duplicated()]


# In[14]:


# Lets visualize boxplots to see how price is getting varied with the factors
#airline vs price

sns.catplot(y = "Price", x = "Airline", data = flight_df.sort_values("Price", ascending = False),kind="boxen", height = 6, aspect = 3)
plt.show()


# In[15]:



#airline vs total stops

sns.catplot(y = "Price", x = "Airline", data = flight_df.sort_values("Price", ascending = False),kind="bar", hue='Total_Stops',height = 6, aspect = 3)
plt.show()


# In[16]:


sns.catplot(y = "Price", x = "Airline", data = flight_df.sort_values("Price", ascending = False),kind="bar", hue='Destination',height = 6, aspect = 3)
plt.show()


# In[17]:


sns.catplot(y = "Price", x = "Airline", data = flight_df.sort_values("Price", ascending = False),kind="bar", hue='Source',height = 6, aspect = 3)
plt.show()


# In[18]:


flight_df.head()


# In[19]:


flight_df['source_destination']=flight_df['Source']+'-'+flight_df['Destination']
flight_df.head()


# In[20]:


sns.catplot(y = "Price", x = "Airline", data = flight_df.sort_values("Price", ascending = False),kind="boxen", hue='source_destination',height = 6, aspect = 3)
plt.show()


# In[21]:


# convert date related columns to datetime format like date_journey, Dep_Time etc
# Date_of_Journey
flight_df["Journey_day"] = pd.to_datetime(flight_df.Date_of_Journey, format="%d/%m/%Y").dt.day
flight_df["Journey_month"] = pd.to_datetime(flight_df["Date_of_Journey"], format = "%d/%m/%Y").dt.month
flight_df.drop(["Date_of_Journey"], axis = 1, inplace = True)

# Dep_Time
flight_df["Dep_hour"] = pd.to_datetime(flight_df["Dep_Time"]).dt.hour
flight_df["Dep_min"] = pd.to_datetime(flight_df["Dep_Time"]).dt.minute
flight_df.drop(["Dep_Time"], axis = 1, inplace = True)

# Arrival_Time
flight_df["Arrival_hour"] = pd.to_datetime(flight_df.Arrival_Time).dt.hour
flight_df["Arrival_min"] = pd.to_datetime(flight_df.Arrival_Time).dt.minute
flight_df.drop(["Arrival_Time"], axis = 1, inplace = True)

# Duration

duration=list(flight_df['Duration']) # convertion duration coulmn to list to sepearte hours and min

for i in range(len(duration)):
    if duration[i].split()!=2:
        if 'h' in duration[i]:
            duration[i]= duration[i].strip()+' 0m'
            
        else:
            duration[i]= '0h '+ duration[i]

duration_hr=[]
duration_min=[]

for i in range(len(duration)):
    duration_hr.append(int(duration[i].split(sep = "h")[0]))    # Extract hours from duration
    duration_min.append(int(duration[i].split(sep = "m")[0].split()[-1]))   # Extracts only minutes from duration

flight_df.drop(["Duration"], axis = 1, inplace = True)


# In[22]:


flight_df.head()


# Since we have the duration and total stops, we can drop the route colun

# In[23]:


# droppping route and source_destination
flight_df.drop(['Route'],axis=1,inplace=True)
flight_df.drop(['source_destination'],axis=1,inplace=True)


# In[24]:


sns.catplot(y = "Price", x = "Airline", data = flight_df.sort_values("Price", ascending = False),kind="bar", hue='Journey_month',height = 6, aspect = 3)
plt.show()


# In[25]:


sns.catplot(y = "Price", x = "Journey_month", data = flight_df.sort_values("Price", ascending = False),kind="bar", hue='Journey_day',height = 6, aspect = 3)
plt.show()


# From the above analysis, following observations could be drawn:
#     * Maximum price is for Jet airways business flights
#     * Majority of the flights with one stops have are more expensive 
#     * All flights prices seem to be maximum in the month of march
#     * Also, prices are higher for flights flying on the 1st day of the month
#     

# There are 6 categorical variables described below:

# In[26]:


flight_df['Airline'].unique()


# In[27]:


flight_df['Source'].unique()


# In[28]:


flight_df['Destination'].unique()


# In[29]:


flight_df['Additional_Info'].unique()


# In[30]:


flight_df['Additional_Info'].value_counts()/flight_df['Additional_Info'].count()


# Almost 80% of the values contain no Info. Hence, we'll be dropping it

# In[31]:


flight_df.drop(['Additional_Info'],axis=1,inplace=True)


# In[32]:


flight_df['Total_Stops'].unique()


# 

# In[33]:


# one-hot encoding of categorical variables
print("Airline")
print("-"*75)
print(flight_df["Airline"].value_counts())
Airline = pd.get_dummies(flight_df["Airline"], drop_first= True)

print(flight_df["Source"].value_counts())
Source = pd.get_dummies(flight_df["Source"], drop_first= True)


print(flight_df["Destination"].value_counts())
Destination = pd.get_dummies(flight_df["Destination"], drop_first = True)


# In[34]:


fight_df = pd.concat([flight_df, Airline, Source, Destination], axis = 1)


# In[35]:


# Replacing Total_Stops
fight_df.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace = True)

fight_df.drop(['Airline','Source','Destination'],axis=1,inplace=True)


# In[36]:


fight_df.head()


# Now that we have created our final dataset, we'll need to divide it into dependant and independant variables and do feture selection

# In[37]:


# X: indpendant variables and Y: dependant variables

X=fight_df.drop(['Price'],axis=1,inplace=False)


# In[38]:


Y=fight_df['Price']


# In[39]:


X.head()


# There are 3 feature selections methods we'll be exploring:
# 1. univariate analysis
# 2. correlation matrix
# 3. feature importance

# In[40]:


# univariate analysis
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

fs=SelectKBest(score_func=chi2)  
# Applying feature selection
X_selected=fs.fit(X,Y)

# graph of features

plt.figure(figsize=(15,15))
feat_importances = pd.Series(X_selected.scores_, index=X.columns)
feat_importances.nlargest(20).plot(kind='barh')
plt.show()


# In[41]:


# correaltion matrix
plt.figure(figsize=(25,25))
sns.heatmap(fight_df.corr(),annot = True, cmap = "RdYlGn")


# In[42]:


# Removing correlated features
Threshold=0.8
# find and remove correlated features
def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr

correlation(X,Threshold)


# In[43]:


# Feature Selection

# Important feature using ExtraTreesRegressor

from sklearn.ensemble import ExtraTreesRegressor
selection = ExtraTreesRegressor()
selection.fit(X, Y)

#plot graph of feature importances for better visualization
plt.figure(figsize = (12,8))
feat_importances = pd.Series(selection.feature_importances_, index=X.columns)
feat_importances.nlargest(20).plot(kind='barh')
plt.show()


# In[44]:


# Making the same changes in test data 

test=pd.read_excel('Test_set.xlsx')
test.isnull().sum()


# In[45]:


test['Source'] = test['Source'].replace({'Delhi': 'New Delhi'})
test['Destination'] = test['Destination'].replace({'Delhi': 'New Delhi'})


# In[46]:


# Date_of_Journey
test["Journey_day"] = pd.to_datetime(test.Date_of_Journey, format="%d/%m/%Y").dt.day
test["Journey_month"] = pd.to_datetime(test["Date_of_Journey"], format = "%d/%m/%Y").dt.month
test.drop(["Date_of_Journey"], axis = 1, inplace = True)

# Dep_Time
test["Dep_hour"] = pd.to_datetime(test["Dep_Time"]).dt.hour
test["Dep_min"] = pd.to_datetime(test["Dep_Time"]).dt.minute
test.drop(["Dep_Time"], axis = 1, inplace = True)

# Arrival_Time
test["Arrival_hour"] = pd.to_datetime(test.Arrival_Time).dt.hour
test["Arrival_min"] = pd.to_datetime(test.Arrival_Time).dt.minute
test.drop(["Arrival_Time"], axis = 1, inplace = True)

# Duration

duration=list(test['Duration']) # convertion duration coulmn to list to sepearte hours and min

for i in range(len(duration)):
    if duration[i].split()!=2:
        if 'h' in duration[i]:
            duration[i]= duration[i].strip()+' 0m'
            
        else:
            duration[i]= '0h '+ duration[i]

duration_hr=[]
duration_min=[]

for i in range(len(duration)):
    duration_hr.append(int(duration[i].split(sep = "h")[0]))    # Extract hours from duration
    duration_min.append(int(duration[i].split(sep = "m")[0].split()[-1]))   # Extracts only minutes from duration

test.drop(["Duration"], axis = 1, inplace = True)


# In[47]:


test.head()


# In[48]:


# one-hot encoding of categorical variables
print("Airline")
print("-"*75)
print(test["Airline"].value_counts())
Airline = pd.get_dummies(test["Airline"], drop_first= True)

print(test["Source"].value_counts())
Source = pd.get_dummies(test["Source"], drop_first= True)


print(test["Destination"].value_counts())
Destination = pd.get_dummies(test["Destination"], drop_first = True)

# Replacing Total_Stops
test.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace = True)

test.drop(['Airline','Source','Destination','Route'],axis=1,inplace=True)


# In[49]:


test.head()


# In[50]:


test = pd.concat([test, Airline, Source, Destination], axis = 1)

test.drop(['Additional_Info'],axis=1,inplace=True)


# In[51]:


test.head()


# # Regression modelling
# I am going to apply 10 regression models and judge them on the basis of RMSE value.
# The models are:
# * LinearRegression
# * LGBM Regressor
# * XGBoost Regressor
# * CatBoost Regressor
# * Stochastic Gradient Descent Regression
# * Kernel Ridge Regression
# * Support Vector Machine
# * Random Forest Regressor
# * Gradient Boosting Regression
# * Bayesian Ridge Regression

# In[52]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)


# In[53]:


# importing ML models
from sklearn.ensemble import RandomForestRegressor


from sklearn.tree import DecisionTreeRegressor

from sklearn.svm import SVR

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.linear_model import BayesianRidge

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import ElasticNet

from xgboost.sklearn import XGBRegressor

from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# In[54]:



models = [['LinearRegression : ', LinearRegression()],
          ['ElasticNet :', ElasticNet()],
          ['DecisionTreeRegressor : ', DecisionTreeRegressor()],
          ['RandomForestRegressor : ', RandomForestRegressor()],
          ['SVR : ', SVR()],
          ['GradientBoostingRegressor : ', GradientBoostingRegressor()],
          ['KNeighborsRegressor : ', KNeighborsRegressor()],
          ['BayesianRidge : ', BayesianRidge()],
          ['KernalRidge: ', KernelRidge()]]


# In[55]:


rmse_score=[]
for name,model in models:
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    rmse_score.append([name,np.sqrt(mean_squared_error(y_test, predictions))])
    print(name, (np.sqrt(mean_squared_error(y_test, predictions)))) 


# In[56]:


rmse_score=sorted(rmse_score,key=lambda x: x[1])
rmse_score


# In[73]:


df=pd.DataFrame(rmse_score,columns=['model','rmse_score'])


# In[77]:


df.head(3)


# Above 3 models have the lowest RMSE. Hence, we will be considering them for hyperparameter tuning

# # HyperParameter Tuning
# 
# * GridSearchCV -> computationally expensive
# * RandomGridSearchCV-> Faster 

# In[80]:


model_para={
    'RandomForestRegressor':{
        'model':RandomForestRegressor(),
        'parameters':{
            'n_estimators' : [300, 500, 700, 1000, 2100],
            'max_depth' : [3, 5, 7, 9, 11, 13, 15],
            'max_features' : ["auto", "sqrt", "log2"],
            'min_samples_split' : [2, 4, 6, 8]
        }
    },
    
    'GradientBoostingRegressor':{
        'model':GradientBoostingRegressor(),
        'parameters':{      
            'learning_rate' : [0.5, 0.8, 0.1, 0.20, 0.25, 0.30],
            'n_estimators' : [300, 500, 700, 1000, 2100],
            'criterion' : ['friedman_mse', 'mse']
        }
    
        
    },
    
    'DecisionTreeRegressor':{
        'model':DecisionTreeRegressor(),
        'parameters':{
            'splitter':["best","random"],
            'max_depth' : [3, 5, 7, 9, 11, 13, 15],
            'max_features' : ["auto", "sqrt", "log2"],
            'min_samples_split' : [2, 4, 6, 8]                 
        }
    }
    

}


# In[96]:


from sklearn.model_selection import RandomizedSearchCV

score = []

for name, mp in model_para.items() :
    rs = RandomizedSearchCV(estimator = mp['model'], param_distributions = mp['parameters'], cv = 5, n_jobs=-1,verbose=2)
    rs.fit(X_train, y_train)
    score.append({
        'model': name,
        'score' : rs.best_score_,
        'params' : rs.best_params_
    })


# In[97]:


df=pd.DataFrame(score)


# In[98]:


df


# In[99]:


df['params'][1]


# In[57]:


best_model=GradientBoostingRegressor(n_estimators= 2100, learning_rate= 0.2, criterion= 'mse')


# In[58]:


best_model.fit(X_train, y_train)
prediction = best_model.predict(X_test)


# In[59]:


best_model.score(X_train, y_train),best_model.score(X_test, y_test)


# In[61]:


plt.figure(figsize = (8,8))
plt.scatter(y_test, prediction, alpha = 0.5)
plt.xlabel("Actual Values")
plt.ylabel("Prdicted Values")
plt.show()


# In[62]:


print('MAE: ',mean_absolute_error(y_test,prediction))
print('MSE: ',mean_squared_error(y_test,prediction))
print('RMSE: ',np.sqrt(mean_squared_error(y_test,prediction)))


# # Saving the model for Web App

# In[63]:


import pickle

pickle.dump(best_model,open('model.pkl','wb'))


# In[64]:


model=pickle.load(open('model.pkl','rb'))


# In[65]:


model.score(X_test,y_test)


# In[ ]:




