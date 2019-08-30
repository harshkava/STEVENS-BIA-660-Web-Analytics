# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 15:11:48 2018

@author: Harsh Kava
"""


import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error,confusion_matrix, precision_score, recall_score, auc,roc_curve
from sklearn import ensemble, linear_model, neighbors, svm, tree, neural_network
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn import svm,model_selection, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from nltk.corpus import stopwords
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

#load package
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from math import sqrt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

movies = pd.read_csv('D:\\Courses\\04_Web Analytics\\Study Material\\Final Project\\sample3.txt',sep='\t',header=0)
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




y = abs (movies['audience_score'] - movies['critic_score'])
mov = movies[['audience_score','critic_score','Runtime']].copy()
seed = 7
scoring = 'accuracy'
validation_size = 0.20

X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(movies, y, test_size=validation_size, random_state=seed)


KNN_classifier=KNeighborsClassifier()
LREG_classifier=LogisticRegression()
DT_classifier = DecisionTreeClassifier()

predictors=[('knn',KNN_classifier),('lreg',LREG_classifier),('dt',DT_classifier)]

VT=VotingClassifier(predictors)

KNN_grid = [{'n_neighbors': [1,3,5,7,9,11,13,15,17], 'weights':['uniform','distance']}]
gridsearchKNN = GridSearchCV(KNN_classifier, KNN_grid, cv=2)
gridsearchKNN.fit(X_train,Y_train)

#=======================================================================================

DT_grid = [{'max_depth': [3,4,5,6,7,8,9,10,11,12],'criterion':['gini','entropy']}]
gridsearchDT  = GridSearchCV(DT_classifier, DT_grid, cv=5)
gridsearchDT.fit(X_train,Y_train)

#=======================================================================================
 

#=======================================================================================

VT.fit(X_train,Y_train)

#use the VT classifier to predict
predicted=VT.predict(X_validation)

#print the accuracy
print (accuracy_score(predicted,Y_validation))






# =============================================================================
X = movies.drop(['RatingDiff'], axis=1)
y = movies[['RatingDiff']]

x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=.25,random_state=1)

MLA = [
    #Ensemble Methods
    ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier(),
    ensemble.ExtraTreesClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(),

    #Gaussian Processes
    gaussian_process.GaussianProcessClassifier(),
    
    #GLM
    linear_model.LogisticRegressionCV(),
    linear_model.PassiveAggressiveClassifier(),
    linear_model. RidgeClassifierCV(),
    linear_model.SGDClassifier(),
    linear_model.Perceptron(),
    
    #Navies Bayes
    naive_bayes.BernoulliNB(),
    naive_bayes.GaussianNB(),
    
    #Nearest Neighbor
    neighbors.KNeighborsClassifier(),
    
    #SVM
    svm.SVC(probability=True),
    svm.NuSVC(probability=True),
    svm.LinearSVC(),
    
    #Trees    
    tree.DecisionTreeClassifier(),
   #tree.ExtraTreeClassifier(),
    ]


MLA_columns = []
MLA_compare = pd.DataFrame(columns = MLA_columns)


row_index = 0
for alg in MLA:
    
    predicted = alg.fit(x_train, y_train).predict(x_test)
    fp, tp, th = roc_curve(y_test, predicted)
    MLA_name = alg.__class__.__name__
    MLA_compare.loc[row_index,'MLA Name'] = MLA_name
    MLA_compare.loc[row_index, 'MLA Train Accuracy'] = round(alg.score(x_train, y_train), 4)
    MLA_compare.loc[row_index, 'MLA Test Accuracy'] = round(alg.score(x_test, y_test), 4)
    MLA_compare.loc[row_index, 'MLA Precission'] = precision_score(y_test, predicted)
    MLA_compare.loc[row_index, 'MLA Recall'] = recall_score(y_test, predicted)
    MLA_compare.loc[row_index, 'MLA AUC'] = auc(fp, tp)

    row_index+=1
    
MLA_compare.sort_values(by = ['MLA Test Accuracy'], ascending = False, inplace = True)    
MLA_compare