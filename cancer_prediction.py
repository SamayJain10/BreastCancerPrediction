#!/usr/bin/env python
# coding: utf-8

# In[1]:


#python modules to import

import numpy as np 
import pandas as pd 
import pickle
import os 
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from xgboost import XGBClassifier
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('darkgrid')


pd.options.display.max_columns = 100


# In[2]:


os.chdir ('C:\\Share\\Samay\\Acmegrade\\Cancer_Prediction\\')
data = pd.read_csv('data.csv')
data


# In[3]:


data.head(10) #shows top 10 entries


# In[4]:


data.tail(10) #shows last 10 entries


# In[5]:


data.dtypes #shows data types of all the header feilds


# In[6]:


data.info


# In[7]:


data.describe(include="O") 


# In[8]:


data.diagnosis.value_counts() #displays count of unique values


# In[9]:


diagnosis_unique = data.diagnosis.unique()


# In[10]:


diagnosis_unique


# In[11]:



plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.hist( data.diagnosis)
#legend()
plt.title("Counts of Diagnosis")
plt.xlabel("Diagnosis")


plt.subplot(1, 2, 2)

sns.countplot('diagnosis', data=data); 



# In[12]:



px.histogram(data, x='diagnosis')
# plt.show()


# In[13]:


cols = ["diagnosis", "radius_mean", "texture_mean", "perimeter_mean", "area_mean"]

sns.pairplot(data[cols], hue="diagnosis")
plt.show()


# In[14]:


from sklearn.preprocessing import LabelEncoder


# In[15]:


data.head()


# In[16]:


labelencoder_Y = LabelEncoder()
data.diagnosis = labelencoder_Y.fit_transform(data.diagnosis)
#converting into numeric values
#M-> 1
#B-> 0


# In[17]:


data.head(10)


# In[18]:


data.tail(10)


# In[19]:


print(data.diagnosis.value_counts())
print("\n", data.diagnosis.value_counts().sum())


# In[20]:


plt.figure(figsize=(12, 9))

plt.title("Correlation Graph")

cmap = sns.diverging_palette( 1000, 120, as_cmap=True)
sns.heatmap(data[cols].corr(), annot=True, fmt='.1%',  linewidths=.5, cmap=cmap);


# In[21]:


plt.figure(figsize=(15, 10))


fig = px.imshow(data[cols].corr());
fig.show()


# In[22]:


size = len(data['texture_mean'])

area = np.pi * (15 * np.random.rand( size ))**2
colors = np.random.rand( size )

plt.xlabel("texture mean")
plt.ylabel("radius mean") 
plt.scatter(data['texture_mean'], data['radius_mean'], s=area, c=colors, alpha=0.5);


# In[23]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


# In[24]:


from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.svm import SVC
from sklearn import metrics


# In[25]:


data.columns


# In[26]:


prediction_feature = [ "radius_mean",  'perimeter_mean', 'area_mean', 'symmetry_mean', 'compactness_mean', 'concave points_mean']

targeted_feature = 'diagnosis'

len(prediction_feature)


# In[27]:


X = data[prediction_feature]
X


# In[28]:


Y = data.diagnosis
Y


# In[29]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=15)


# In[30]:


print(X_train)


# In[31]:


sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# In[32]:


def model_building(model, X_train, X_test, y_train, y_test):
    """
    
    Model Fitting, Prediction And Other stuff
    return ('score', 'accuracy_score', 'predictions' )
    """
    
    model.fit(X_train, y_train)
    score = model.score(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(predictions, y_test)
    
    return (score, accuracy, predictions)    


# In[33]:


models_list = {
    "LogisticRegression" :  LogisticRegression(),
    "RandomForestClassifier" :  RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=5),
    "DecisionTreeClassifier" :  DecisionTreeClassifier(criterion='entropy', random_state=0),
    "SVC" :  SVC(),
}


# In[34]:


print(list(models_list.keys()))
print(list(models_list.values()))


# In[35]:


def cm_metrix_graph(cm):
    
    sns.heatmap(cm,annot=True,fmt="d")
    plt.show()
        


# In[36]:


df_prediction = []
confusion_matrixs = []
df_prediction_cols = [ 'model_name', 'score', 'accuracy_score' , "accuracy_percentage"]

for name, model in zip(list(models_list.keys()), list(models_list.values())):
    
    (score, accuracy, predictions) = model_building(model, X_train, X_test, Y_train, Y_test )
    
    print("\n\nClassification Report of '"+ str(name), "'\n")
    
    print(classification_report(Y_test, predictions))

    df_prediction.append([name, score, accuracy, "{0:.2%}".format(accuracy)])
    
    # For Showing Metrics
    confusion_matrixs.append(confusion_matrix(Y_test, predictions))
    
df_pred = pd.DataFrame(df_prediction, columns=df_prediction_cols)    


# In[37]:


print(len(confusion_matrixs))


# In[38]:


df_pred


# In[39]:


df_pred.sort_values('score', ascending=False)


# In[40]:


def cross_val_scorring(model):
    
#     (score, accuracy, predictions) = model_building(model, X_train, X_test, y_train, y_test )
    
    model.fit(data[prediction_feature], data[targeted_feature])
    
    # score = model.score(X_train, y_train)    
    
    predictions = model.predict(data[prediction_feature])    
    accuracy = accuracy_score(predictions, data[targeted_feature])
    print("\nFull-Data Accuracy:", round(accuracy, 2))
    print("Cross Validation Score of'"+ str(name), "'\n")
    
    
    # Initialize K folds.
    kFold = KFold(n_splits=5) # define 5 diffrent data folds
    
    err = []
    
    for train_index, test_index in kFold.split(data):
        # print("TRAIN:", train_index, "TEST:", test_index)

        # Data Spliting via fold indexes
        X_train = data[prediction_feature].iloc[train_index, :] # train_index = rows and all columns for Prediction_features
        y_train = data[targeted_feature].iloc[train_index] # all targeted features trains
        
        X_test = data[prediction_feature].iloc[test_index, :] # testing all rows and cols
        y_test = data[targeted_feature].iloc[test_index] # all targeted tests
        
        # Again Model Fitting
        model.fit(X_train, y_train)

        err.append(model.score(X_train, y_train))
        
        print("Score:", round(np.mean(err),  2) )


# In[41]:


for name, model in zip(list(models_list.keys()), list(models_list.values())):
    cross_val_scorring(model)


# In[42]:


from  sklearn.model_selection import GridSearchCV


# In[43]:


# Pick the model
model = DecisionTreeClassifier()

# Tunning Params
param_grid = {'max_features': ['auto', 'sqrt', 'log2'],
              'min_samples_split': [2,3,4,5,6,7,8,9,10], 
              'min_samples_leaf':[2,3,4,5,6,7,8,9,10] }


# Implement GridSearchCV
gsc = GridSearchCV(model, param_grid, cv=10) # For 10 Cross-Validation

gsc.fit(X_train, Y_train) # Model Fitting

print("\n Best Score is ")
print(gsc.best_score_)

print("\n Best Estimator is ")
print(gsc.best_estimator_)

print("\n Best Parametes are")
print(gsc.best_params_)


# In[44]:


# Pick the model
model = KNeighborsClassifier()


# Tunning Params
param_grid = {
    'n_neighbors': list(range(1, 30)),
    'leaf_size': list(range(1,30)),
    'weights': [ 'distance', 'uniform' ]
}


# Implement GridSearchCV
gsc = GridSearchCV(model, param_grid, cv=10)

# Model Fitting
gsc.fit(X_train, Y_train)

print("\n Best Score is ")
print(gsc.best_score_)

print("\n Best Estimator is ")
print(gsc.best_estimator_)

print("\n Best Parametes are")
print(gsc.best_params_)


# In[45]:


# Pick the model
model = SVC()


# Tunning Params
param_grid = [
              {'C': [1, 10, 100, 1000], 
               'kernel': ['linear']
              },
              {'C': [1, 10, 100, 1000], 
               'gamma': [0.001, 0.0001], 
               'kernel': ['rbf']
              }
]


# Implement GridSearchCV
gsc = GridSearchCV(model, param_grid, cv=10) # 10 Cross Validation

# Model Fitting
gsc.fit(X_train, Y_train)

print("\n Best Score is ")
print(gsc.best_score_)

print("\n Best Estimator is ")
print(gsc.best_estimator_)

print("\n Best Parametes are")
print(gsc.best_params_)


# In[46]:


# Pick the model
model = RandomForestClassifier()


# Tunning Params
random_grid = {'bootstrap': [True, False],
 'max_depth': [40, 50, None], # 10, 20, 30, 60, 70, 100,
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2], # , 4
 'min_samples_split': [2, 5], # , 10
 'n_estimators': [200, 400]} # , 600, 800, 1000, 1200, 1400, 1600, 1800, 2000

# Implement GridSearchCV
gsc = GridSearchCV(model, random_grid, cv=10) # 10 Cross Validation

# Model Fitting
gsc.fit(X_train, Y_train)

print("\n Best Score is ")
print(gsc.best_score_)

print("\n Best Estimator is ")
print(gsc.best_estimator_)

print("\n Best Parametes are")
print(gsc.best_params_)


# In[47]:


import pickle as pkl


# In[48]:


# Trainned Model # You can also use your own trainned model
logistic_model = LogisticRegression()
logistic_model.fit(X_train, Y_train)

filename = 'logistic_model.pkl'
pkl.dump(logistic_model, open(filename, 'wb')) # wb means write as binary


# In[49]:


loaded_model = pkl.load(open(filename, 'rb')) # rb means read as binary
result = loaded_model.score(X_test, Y_test)
print(result)


# In[50]:


load_model =pickle.load(open('C:\Share\Samay\Acmegrade\Cancer_Prediction\logistic_model.pkl','rb'))


# In[51]:


pred1 = load_model.predict (X_test)
pred1


# In[52]:


print (accuracy_score (pred1,Y_test))


# In[ ]:




