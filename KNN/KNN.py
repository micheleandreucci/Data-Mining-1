#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np  
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns


# In[2]:


df = pd.read_csv("HR_Employee_MissingValuesFilled.csv") 
df.head()


# In[3]:


df.info()


# In[4]:


plt.figure(figsize =(10, 4)) 
sns.heatmap(df.isnull(), yticklabels = False, cbar = False, cmap ='viridis')


# In[5]:


sns.set_style('darkgrid') 
sns.countplot(x ='Attrition', data = df) 


# In[7]:


sns.lmplot(x = 'Age', y = 'DailyRate', hue = 'Attrition', data = df) 


# In[8]:


plt.figure(figsize =(10, 6)) 
sns.boxplot(y ='MonthlyIncome', x ='Attrition', data = df)


# In[9]:


to_discard = ['Over18','StandardHours','TrainingTimesLastYear']
to_df = [col for col in df.columns if col not in to_discard]
df=df[to_df]


# In[10]:


y = df.iloc[:, 1] 
X = df 
X.drop('Attrition', axis = 1, inplace = True) 


# In[11]:


from sklearn.preprocessing import LabelEncoder 
lb = LabelEncoder() 
y = lb.fit_transform(y) 


# In[12]:


dum_BusinessTravel = pd.get_dummies(df['BusinessTravel'],  
                                    prefix ='BusinessTravel') 
dum_Department = pd.get_dummies(df['Department'],  
                                prefix ='Department') 
dum_EducationField = pd.get_dummies(df['EducationField'],  
                                    prefix ='EducationField') 
dum_Gender = pd.get_dummies(df['Gender'],  
                            prefix ='Gender', drop_first = True) 
dum_JobRole = pd.get_dummies(df['JobRole'],  
                             prefix ='JobRole') 
dum_MaritalStatus = pd.get_dummies(df['MaritalStatus'],  
                                   prefix ='MaritalStatus') 
dum_OverTime = pd.get_dummies(df['OverTime'],  
                              prefix ='OverTime', drop_first = True) 
# Adding these dummy variable to input X 
X = pd.concat([X, dum_BusinessTravel, dum_Department,  
               dum_EducationField, dum_Gender, dum_JobRole,  
               dum_MaritalStatus, dum_OverTime], axis = 1) 
# Removing the categorical data 
X.drop(['BusinessTravel', 'Department', 'EducationField',  
        'Gender', 'JobRole', 'MaritalStatus', 'OverTime'],  
        axis = 1, inplace = True) 
  
print(X.shape) 
print(y.shape) 


# In[13]:


from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split( 
    X, y, test_size = 0.25, random_state = 40)


# In[14]:


from sklearn.neighbors import KNeighborsClassifier 
neighbors = []  
cv_scores = []  
    
from sklearn.model_selection import cross_val_score  
# perform 10 fold cross validation  
for k in range(1, 40, 2):  
    neighbors.append(k)  
    knn = KNeighborsClassifier(n_neighbors = k)  
    scores = cross_val_score(  
        knn, X_train, y_train, cv = 10, scoring = 'accuracy')  
    cv_scores.append(scores.mean()) 
error_rate = [1-x for x in cv_scores]  
    
# determining the best k  
optimal_k = neighbors[error_rate.index(min(error_rate))]  
print('The optimal number of neighbors is % d ' % optimal_k)  
    
# plot misclassification error versus k  
plt.figure(figsize = (10, 6))  
plt.plot(range(1, 40, 2), error_rate, color ='blue', linestyle ='dashed', marker ='o', 
         markerfacecolor ='red', markersize = 10) 
plt.xlabel('Number of neighbors')  
plt.ylabel('Misclassification Error')  
plt.show()  


# In[15]:


from sklearn.model_selection import cross_val_predict, cross_val_score 
from sklearn.metrics import accuracy_score, classification_report 
from sklearn.metrics import confusion_matrix 
  
def print_score(clf, X_train, y_train, X_test, y_test, train = True): 
    if train: 
        print("Train Result:") 
        print("------------") 
        print("Classification Report: \n {}\n".format(classification_report( 
                y_train, clf.predict(X_train)))) 
        print("Confusion Matrix: \n {}\n".format(confusion_matrix( 
                y_train, clf.predict(X_train)))) 
  
        res = cross_val_score(clf, X_train, y_train,  
                              cv = 10, scoring ='accuracy') 
        print("Average Accuracy: \t {0:.4f}".format(np.mean(res))) 
        print("Accuracy SD: \t\t {0:.4f}".format(np.std(res))) 
        print("accuracy score: {0:.4f}\n".format(accuracy_score( 
                y_train, clf.predict(X_train)))) 
        print("----------------------------------------------------------") 
                 
    elif train == False: 
        print("Test Result:") 
        print("-----------") 
        print("Classification Report: \n {}\n".format( 
                classification_report(y_test, clf.predict(X_test)))) 
        print("Confusion Matrix: \n {}\n".format( 
                confusion_matrix(y_test, clf.predict(X_test))))  
        print("accuracy score: {0:.4f}\n".format( 
                accuracy_score(y_test, clf.predict(X_test)))) 
        print("-----------------------------------------------------------") 


# In[16]:


knn = KNeighborsClassifier(n_neighbors = 7) 
knn.fit(X_train, y_train) 
print_score(knn, X_train, y_train, X_test, y_test, train = True) 
print_score(knn, X_train, y_train, X_test, y_test, train = False) 


# In[17]:


knn = KNeighborsClassifier(n_neighbors = 17) 
knn.fit(X_train, y_train) 
print_score(knn, X_train, y_train, X_test, y_test, train = True) 
print_score(knn, X_train, y_train, X_test, y_test, train = False) 


# In[ ]:




