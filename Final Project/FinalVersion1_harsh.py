# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 19:42:45 2018

@author: Harsh Kava
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import nan
from scipy import spatial
import operator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import csv
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from nltk.corpus import stopwords
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cross_validation import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import  BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.model_selection import KFold
import warnings

warnings.filterwarnings('ignore')

movies = pd.read_csv('D:\\Courses\\04_Web Analytics\\Study Material\\Final Project\\sample.txt',sep='\t',header=0)
movies.shape
list(movies)

def convertPercentagetoNumber(x):
    x= str(x)
    x = x.replace("%", "")
    return float(x)
    
def convertRuntimetoNumber(x):
    x= str(x)
    x = x.split(" ")[0]
    return float(x)

def convertCurrencytoNumber(x):
    x= str(x)
    x = x.replace("$", "").replace(",", "").replace(" ", "")
    return float(x)


movies['audience_score'] = movies['audience_score'].replace('No Score Yet', '0%')
movies['audience_score'] = movies['audience_score'].apply(convertPercentagetoNumber)
movies['audience_score'].fillna(movies['audience_score'].median(axis=0),inplace=True)
        

movies['critic_score'].replace('No Score Yet', '0%')
movies['critic_score'] = movies['critic_score'].apply(convertPercentagetoNumber)
movies['critic_score'].fillna(movies['critic_score'].median(axis=0), inplace=True)
  
movies['Runtime'] = movies['Runtime'].apply(convertRuntimetoNumber)
movies['Runtime'].fillna(movies['Runtime'].median(axis=0), inplace=True)

movies['Box Office'] = movies['Box Office'].replace('NONE', '0')
movies['Box Office'] = movies['Box Office'].replace('NA', '0')
movies['Box Office'] = movies['Box Office'].apply(convertCurrencytoNumber)
movies['Box Office'].fillna(0, inplace=True) 

movies['RatingDiff'] = abs (movies['audience_score'] - movies['critic_score'])
  

genreVectorizor = TfidfVectorizer(lowercase=True, norm=None, stop_words='english', use_idf=False)
g= genreVectorizor.fit_transform(movies['Genre'].values.astype('U')).toarray()

df1 = pd.DataFrame(g, columns=genreVectorizor.get_feature_names())
frames = [movies, df1]
movies = pd.concat(frames,axis=1, join_axes=[movies.index]) #,columns=genreVectorizor.get_feature_names())
actor_list_per_movie = list(map(str,(movies['actor_names'])))
actorSet = set()
for i in actor_list_per_movie:
    split_actor = list(map(str, i.split(',')))
    for j in split_actor:
        actorSet.add(j.lower().strip())  
actorNamesVectorizor = CountVectorizer(stop_words='english', vocabulary = actorSet)
x= actorNamesVectorizor.fit_transform(movies['actor_names'].values.astype('U')).toarray()

df1 = pd.DataFrame(x, columns=actorNamesVectorizor.get_feature_names())
frames = [movies, df1]

movies = pd.concat(frames,axis=1, join_axes=[movies.index])
director_list_per_movie = list(map(str,(movies['Directed By'])))
directorSet = set()
for i in director_list_per_movie:
    split_director = list(map(str, i.split(',')))
    for j in split_director:
        directorSet.add(j.lower().strip())  
directorNamesVectorizor = CountVectorizer(stop_words='english', vocabulary = directorSet)
g= directorNamesVectorizor.fit_transform(movies['Directed By'].values.astype('U')).toarray()
df1 = pd.DataFrame(g, columns=directorNamesVectorizor.get_feature_names())
frames = [movies, df1]
movies = pd.concat(frames,axis=1, join_axes=[movies.index])


studio_list_per_movie = list(map(str,(movies['Studio'])))
studioSet = set()
for i in studio_list_per_movie:
    split_studio = list(map(str, i.split(',')))
    for j in split_studio:
        studioSet.add(j.lower().strip())  
studioNamesVectorizor = CountVectorizer(stop_words='english', vocabulary = studioSet)
x= studioNamesVectorizor.fit_transform(movies['Studio'].values.astype('U')).toarray()
df1 = pd.DataFrame(x, columns=studioNamesVectorizor.get_feature_names())
frames = [movies, df1]


movies = pd.concat(frames,axis=1, join_axes=[movies.index])
writer_list_per_movie = list(map(str,(movies['Written By'])))
writerSet = set()
for i in writer_list_per_movie:
    split_writer = list(map(str, i.split(',')))
    for j in split_writer:
        writerSet.add(j.lower().strip())  
writerNamesVectorizor = CountVectorizer(stop_words='english', vocabulary = writerSet)
x= writerNamesVectorizor.fit_transform(movies['Written By'].values.astype('U')).toarray()
df1 = pd.DataFrame(x, columns=writerNamesVectorizor.get_feature_names())
frames = [movies, df1]
movies = pd.concat(frames,axis=1, join_axes=[movies.index])


movies = movies.drop(['movie_id','actor_names','actor_links','synopsis','In Theaters','Genre','Studio','Directed By','Rating','Written By'],1)
movies = movies.fillna(0)

# =============================================================================
X = movies.drop(["audience_score",'critic_score'], axis=1)
y = movies[["audience_score",'critic_score']]
# 
# =============================================================================

# =============================================================================
# x = movies.drop(['critic_score'], axis=1)
# y = movies[['critic_score']]
# =============================================================================

#y = abs (movies['audience_score'] - movies['critic_score'])
#mov = movies[['audience_score','critic_score','Runtime']].copy()
seed = 0
scoring = 'accuracy'
validation_size = 0.30

X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, y, test_size=0.2, random_state=123)


# # # Spot Check Algorithms
models = []
#models.append(('LR', LogisticRegression()))
#models.append(('LDA', LinearDiscriminantAnalysis()))
#models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
#models.append(('BNB', BernoulliNB()))
#models.append(('RF', RandomForestClassifier(n_estimators=2500, n_jobs=15,criterion="entropy",max_features='log2',random_state=150,max_depth=600,min_samples_split=163)))
#models.append(('GBM', AdaBoostClassifier()))
#models.append(('MLP', MLPClassifier()))
#models.append(('SVC', SVC()))
#models.append(('RF', RandomForestClassifier()))


# # # evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(DecisionTreeClassifier(), X_train,Y_train, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)






'''
clf = RandomForestClassifier(n_jobs=2, random_state=0)

# Train the Classifier to take the training features and learn how they relate
# to the training y (the species)
clf.fit(x, y)

preds = clf.predict(y)

pd.crosstab(movies, preds)


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)

print(X_train.shape, y_train.shape)

print(X_test.shape, y_test.shape)


clf = SVC(kernel='linear', C=1).fit(X_train, y_train)
print(clf.score(X_test, y_test))



import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import sklearn

from pandas import Series, DataFrame
from pylab import rcParams
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics 
from sklearn.metrics import classification_report


sb.heatmap(movies.corr())  
movies['RatingDiff']

movies.info()
x = movies.drop(['RatingDiff'], axis=1).values
y = movies[['RatingDiff']].values

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = .3, random_state=25)

clf = LogisticRegression()
clf.fit(X_train,y_train)

clf.predict(X_test)

y_pred 
'''