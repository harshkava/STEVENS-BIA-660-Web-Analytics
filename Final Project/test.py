# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 18:07:16 2018

@author: Harsh Kava
"""

df = loan_2.reindex(columns= ['term_clean','grade_clean', 'annual_inc', 'loan_amnt', 'int_rate','purpose_clean','installment','loan_status_clean'])
df.fillna(method= 'ffill').astype(int)
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
imp = Imputer(missing_values='NaN', strategy='median', axis=0)
array = df.values
y = df['loan_status_clean'].values
imp.fit(array)
array_imp = imp.transform(array)

y2= y.reshape(1,-1)
imp.fit(y2)
y_imp= imp.transform(y2)
X = array_imp[:,0:4]
Y = array_imp[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
seed = 7
scoring = 'accuracy'

from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import  BernoulliNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier


from sklearn.svm import SVC
from sklearn.svm import SVR
# Spot Check Algorithms



df = loan_2.reindex(columns= ['term_clean','grade_clean', 'annual_inc', 'loan_amnt', 'int_rate','purpose_clean','installment','loan_status_clean'])
df.fillna(method= 'ffill').astype(int)
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
imp = Imputer(missing_values='NaN', strategy='median', axis=0)
array = df.values
y = df['loan_status_clean'].values
imp.fit(array)
array_imp = imp.transform(array)

y2= y.reshape(1,-1)
imp.fit(y2)
y_imp= imp.transform(y2)
X = array_imp[:,0:4]
Y = array_imp[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
seed = 7
scoring = 'accuracy'



models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('BNB', BernoulliNB()))
models.append(('RF', RandomForestClassifier()))
models.append(('GBM', AdaBoostClassifier()))
models.append(('NN', MLPClassifier()))
models.append(('SVM', SVC()))
models.append(('SVM', SVR()))

# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)