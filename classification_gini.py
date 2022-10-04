#%matplotlib inline
import graphviz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.constants import lb

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.metrics import roc_curve, auc, roc_auc_score

from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

import pydotplus
from sklearn import tree
from IPython.display import Image

import os
os.environ['PATH'] += os.pathsep + 'C:/Users/saverio/Desktop/Data Mining/DataMiningProject/venvv/Lib/site-packages/graphviz/bin'

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


def groupDistanceFromHome(data):
    if int(data) >=1 and int(data) <= 5:
        return 'near'
    elif int(data) >=6 and int(data) <= 15:
        return 'medium'
    else:
        return 'far'


def groupYearsInCurrentRole(data):
    if int(data) >= 0 and int(data) <= 4:
        return 'short'
    elif int(data) > 4 and int(data) <= 9:
        return 'medium'
    else:
        return 'long'


def groupYearsWithCurrManager(data):
    if int(data) >= 0 and int(data) <= 3:
        return 'short'
    elif int(data) > 3 and int(data) <= 8:
        return 'medium'
    else:
        return 'long'


def groupYearsSinceLastPromotion(data):
    if int(data) >= 0 and int(data) <= 3:
        return 'short'
    elif int(data) > 3 and int(data) <= 8:
        return 'medium'
    else:
        return 'long'


def groupYearsAtCompany(data):
    if int(data) >= 0 and int(data) <= 5:
        return 'short'
    elif int(data) > 5 and int(data) <= 12:
        return 'medium'
    else:
        return 'long'


df = pd.read_csv("HR_Employee_MissingValuesFilled.csv", skipinitialspace=True, sep=',')


df['DistanceFromHome'] = df['DistanceFromHome'].apply(lambda row: groupDistanceFromHome(row))
df['YearsInCurrentRole'] = df['YearsInCurrentRole'].apply(lambda row: groupYearsInCurrentRole(row))
df['YearsWithCurrManager'] = df['YearsWithCurrManager'].apply(lambda row: groupYearsWithCurrManager(row))
df['YearsSinceLastPromotion'] = df['YearsSinceLastPromotion'].apply(lambda row: groupYearsSinceLastPromotion(row))
df['YearsAtCompany'] = df['YearsAtCompany'].apply(lambda row: groupYearsAtCompany(row))


print(df['Attrition'].value_counts())

#create binary variables
cat_columns = ['BusinessTravel', 'Department','Education', 'EducationField', 'JobRole', 'MaritalStatus', 'Gender', 'DistanceFromHome', 'YearsInCurrentRole', 'YearsWithCurrManager', 'YearsSinceLastPromotion', 'YearsAtCompany']
categorical_columns = pd.get_dummies(df[cat_columns])
print(categorical_columns)

column2drop = ['Over18', 'StandardHours', 'TrainingTimesLastYear', 'BusinessTravel', 'Department','Education', 'EducationField', 'JobRole', 'MaritalStatus', 'Gender', 'DistanceFromHome', 'YearsInCurrentRole', 'YearsWithCurrManager', 'YearsSinceLastPromotion', 'YearsAtCompany']
df.drop(column2drop, axis=1, inplace=True)

df = pd.concat([df, categorical_columns], axis=1)

print(df.head())
print(df.info())


le = LabelEncoder()
num_classes = le.fit_transform(df['Attrition'])


#feature reshaping
label_encoders = dict()
column2encode = ['OverTime']

for col in column2encode:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le


#split dataset train and set
attributes = [col for col in df.columns if col != 'Attrition']
X = df[attributes].values
y = df['Attrition']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=100,
                                                    stratify=y)

print(X_train.shape, X_test.shape)

#tuning hyperparam with randomize search

#This has two main benefits over an exhaustive search:

#A budget can be chosen independent of the number of parameters and possible values.

#Adding parameters that do not influence the performance does not decrease efficiency.
#RANDOM


print("Parameter Tuning gini measure: \n")
param_list = {'max_depth': [None] + list(np.arange(2, 20)),
              'min_samples_split': [2, 3, 5, 7, 10, 15, 20, 25, 30, 50, 75, 100],
              'min_samples_leaf': [1, 3, 5, 7, 10, 15, 20, 25, 30, 50, 75, 100],
             }

clf = DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1)

random_search = RandomizedSearchCV(clf, param_distributions=param_list, n_iter=100)
random_search.fit(X, y)
report(random_search.cv_results_, n_top=3)


#build a model
clf = DecisionTreeClassifier(criterion='gini',
                             min_samples_split=15, min_samples_leaf=15)
#clf = DecisionTreeClassifier(criterion='gini', max_depth=7,
#                             min_samples_split=5, min_samples_leaf=20)
clf.fit(X_train, y_train)

#value importance
for col, imp in zip(attributes, clf.feature_importances_):
    print(col, imp)


#visualize the tree
dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=attributes,
                                class_names=clf.classes_,
                                filled=True, rounded=True,
                                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())

graph2 = graphviz.Source(dot_data)
graph2.format = "png"
graph2.render("gini_tree_2")

#Apply the decision tree on the training set
print("Apply the decision tree on the training set: \n")
y_pred = clf.predict(X_train)
print('Accuracy %s' % accuracy_score(y_train, y_pred))
print('F1-score %s' % f1_score(y_train, y_pred, average=None))

print(classification_report(y_train, y_pred))

confusion_matrix(y_train, y_pred)

#Apply the decision tree on the test set and evaluate the performance
print("Apply the decision tree on the test set and evaluate the performance: \n")
y_pred = clf.predict(X_test)

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred))
confusion_matrix(y_test, y_pred)





#ROC CURVE

y_test = le.fit_transform(y_test)

y_pred = le.fit_transform(y_pred)

print("ROC CURVE")
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
print("Roc Curve precision")
print(roc_auc)

plt.figure(figsize=(8, 5))
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % (roc_auc))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=22)
plt.legend(loc="lower right", fontsize=14, frameon=False)
plt.show()
