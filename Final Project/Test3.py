# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 19:14:09 2018

@author: Harsh Kava
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 19:52:19 2018

@author: Harsh Kava
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import nan
from scipy import spatial
import operator
from sklearn.feature_extraction.text import TfidfVectorizer
import csv

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

from sklearn.ensemble import AdaBoostClassifier
from sklearn.cross_validation import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

#movies = pd.read_csv('D:\\Courses\\04_Web Analytics\\Study Material\\Final Project\\Our_movies.txt',sep='\t',
#                   decimal='.',header=0,encoding = "ISO-8859-1")
#pandas.read_csv(filename, sep='\t', lineterminator='\r')
#movies = pd.read_csv('D:\\Courses\\04_Web Analytics\\Study Material\\Final Project\\Our_movies.txt',sep='\t',header=0)

movies = pd.read_csv('D:\\Courses\\04_Web Analytics\\Study Material\\Final Project\\sample.txt',sep='\t',header=0)


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

def getSimilarity(movieId1, movieId2,arg1,arg2):
    a = movies.iloc[movieId1]
    b = movies.iloc[movieId2]
    
    RatingA = a[arg1]
    RatingB = b[arg2]
    
    RatingDistance = spatial.distance.cosine(RatingA, RatingB)
    return RatingDistance

def calculatecore(index, column):
    new_movie=movies.iloc[index].to_frame().T
    
    def getRatingNeighbors(baseMovie, K):
        distances = []
    
        for index, movie in movies.iterrows():
            if movie['new_id'] != baseMovie['new_id'].values[0]:
                dist = getSimilarity(baseMovie['new_id'].values[0], movie['new_id'],column,column)
                distances.append((movie['new_id'], dist))
    
        distances.sort(key=operator.itemgetter(1))
        neighbors = []
    
        for x in range(K):
            neighbors.append(distances[x])
        return neighbors

    K = 10
    avgRating = 0
    neighbors = getRatingNeighbors(new_movie, K)

    for neighbor in neighbors:
        avgRating = avgRating+sum(movies.iloc[neighbor[0]][column]) 
    
    avgRating = avgRating/K
    return avgRating


#movies.describe()

#movies.head(10)

#movies.info()
#Handling outliers before updating missing Values
    
def cleanAudienceScore(movies):
    try:
        movies['audience_score'] = movies['audience_score'].replace('No Score Yet', '0%')
        movies['audience_score'] = movies['audience_score'].apply(convertPercentagetoNumber)
        movies['audience_score'].fillna(movies['audience_score'].median(axis=0),inplace=True)
        #print(movies['audience_score'])
    except Exception as e:
        print('Exception in cleaning Audience_Score column')
        print('Exception is ::',e)


def cleanCriticScore(movies):
    try:
        movies['critic_score'].replace('No Score Yet', '0%')
        movies['critic_score'] = movies['critic_score'].apply(convertPercentagetoNumber)
        movies['critic_score'].fillna(movies['critic_score'].median(axis=0), inplace=True)
       # print(movies['critic_score'])
    except Exception as e:
        print('Exception in cleaning Critic_Score column')
        print('Exception is ::',e)

def cleanRunTime(movies):
    try:
        #movies['Runtime'] = movies['Runtime'].replace('NA', '0')
        movies['Runtime'] = movies['Runtime'].apply(convertRuntimetoNumber)
        movies['Runtime'].fillna(movies['Runtime'].median(axis=0), inplace=True)
        #print(movies['Runtime'])
    except Exception as e:
        print('Exception in cleaning Runtime column')
        print('Exception is ::',e)
        
def cleanBoxOffice(movies):        
    try:
        movies['Box Office'] = movies['Box Office'].replace('NONE', '0')
        movies['Box Office'] = movies['Box Office'].replace('NA', '0')
        movies['Box Office'] = movies['Box Office'].apply(convertCurrencytoNumber)
        movies['Box Office'].fillna(0, inplace=True) 
        #print( movies['Box Office'])
    except Exception as e:
        print('Exception in cleaning Runtime  column')
        print('Exception is ::',e)


def calculateRatingDifference(movies):
    try:
        movies['RatingDiff'] = abs (movies['audience_score'] - movies['critic_score'])
    except Exception as e:
        print('Exception in calculateRatingDifference')
        print('Exception is ::',e)    

#Genre
def createGenreMatrix(movies):    
    genrelist = []
    try:
        genre_details = list(map(str,(movies['Genre'])))

        for i in genre_details:
            split_genre = list(map(str, i.split(',')))
            for j in split_genre:
                if j not in genrelist:
                    genrelist.append(j)
        
        genrelist = set(genrelist)
    except Exception as e:
        print('Exception in creating distinct GenreList')
        print('Exception is ::',e)
    
    def binaryGenrelist(genre_list):
        binaryList = []       
        #print(type(genre_list))
        for genre in genrelist:
            if genre in genre_list:
                binaryList.append(1)
            else:
                binaryList.append(0)
        
        return binaryList
    
    try:
        movies['genre_bin'] = movies['Genre'].astype('str').apply(binaryGenrelist)
    except Exception as e:
        print('Exception in creating BinaryGenreList')
        print('Exception is ::',e)

def calculateGenreScore(movies):
    try:
        genrescores = []
        for i in range(0,movies.shape[0]):
            genrescore = calculatecore(i,'genre_bin')
            genrescores.append(genrescore)
        movies['genrescores'] = genrescores
        
    except Exception as e:
        print('Exception in calculateGenreScore')
        print('Exception is ::',e) 

def createActorMatrix(movies):
    
    #creating actor matrix using actor facebook likes
    actor_likes = {}
    try:
        with open('actor_rating.csv',encoding="utf8") as csvfile:  
            reader = csv.DictReader(csvfile)
            for row in reader:
                actor_likes[row['actor_1_name']] = row['actor_1_facebook_likes']
        
    except Exception as e:
        print('Exception in creating actor <-> FB_likes dictionary from csv')
        print('Exception is ::',e)
        
    
    try:
        if len(actor_likes) > 0 :
            '''
            actor_list_per_movie = list(map(str,(movies['actor_names'])))
            #actor_list_per_movie
            actorSet = set()
            for i in actor_list_per_movie:
                split_actor = list(map(str, i.split(',')))
                for j in split_actor:
                    actorSet.add(j.lower())
            #distinctActorList = set(actorlist)
            f = open ('ActorList.txt','w')
            for x in actorSet:
                f.write(x +'\n')
            
            f.close()
            '''
            def actorScore(actorlist):
                actorScores = []
                #print(actorlist)
                actors = actorlist.split(',')
                
                for a in actors:
                    #print(a)
                    if a in actor_likes:
                        #print('Match in FB List')
                        actorScores.append(actor_likes[a])
                    else:
                        #print('Not in FB List')
                        actorScores.append(0)

                return actorScores
            
            movies['actor_bin'] = movies['actor_names'].astype('str').apply(actorScore)
    except Exception as e:
        print('Exception in creating actor Matrix ')
        print('Exception is ::',e)
        
def calculateActorsScore(movies):
    try:    
        actorscores = []
        for i in range(0,movies.shape[0]):
            actorscore = calculatecore(i,'actor_bin')
            actorscores.append(actorscore)
        movies['actorscores'] = actorscores        
    except Exception as e:
        print('Exception in creating actor Scores column ')
        print('Exception is ::',e)   

#Directed By
def createDirectorMatrix(movies):

#Can uncommment this section, we Don't want to use Director Likes from FB to calcualte score
    '''
    director_details = list(map(str,(movies['Directed By'])))
    directorlist = []
    for i in director_details:
        split_director = list(map(str, i.split(',')))
        for j in split_director:
            if j not in directorlist:
                directorlist.append(j)
    def binary(director_list):
        binaryList = []
        
        for director in directorlist:
            if director in director_list:
                binaryList.append(1)
            else:
                binaryList.append(0)
        
        return binaryList
    movies['director_bin'] = movies['Directed By'].apply(lambda x: binary(x))
    '''    
    #creating actor matrix using actor facebook likes
    director_likes = {}
    try:
        with open('director_rating.csv',encoding="utf8") as csvfile:  
            reader = csv.DictReader(csvfile)
            for row in reader:
                director_likes[str(row['director_name']).lower()] = row['director_facebook_likes']
    except Exception as e:
        print('Exception in creating director <-> FB_likes dictionary from csv')
        print('Exception is ::',e)

    try:
        if len(director_likes) > 0 :
            '''
            director_list_per_movie = list(map(str,(movies['Directed By'])))
            directorSet = set()
            for i in director_list_per_movie:
                split_director = list(map(str, i.split(',')))
                for j in split_director:
                    directorSet.add(j.lower().strip())
            f = open ('directorList.txt','w')
            for x in directorSet:
                f.write(x +'\n')
            f.close()
            '''
            def directorScore(directorlist):
                directorScores = []
                #print(directorlist)
                directors = directorlist.split(',')
                
                for a in directors:
                    #print(a)
                    if a.lower() in director_likes:
                        #print('Match in FB List')
                        directorScores.append(director_likes[a.lower()])
                    else:
                        #print('Not in FB List')
                        directorScores.append(0)
                return directorScores
            
            movies['director_bin'] = movies['Directed By'].astype('str').apply(directorScore)
    except Exception as e:
        print('Exception in creating director Matrix ')
        print('Exception is ::',e)
        

def calculateDirectorScore(movies):
    try:
        directorscores = []
        for i in range(0,movies.shape[0]):
            directorscore = calculatecore(i,'director_bin')
            directorscores.append(directorscore)
        movies['directorscores'] = directorscores
        
    except Exception as e:
            print('Exception in calculating Director Scores')
            print('Exception is ::',e)   

#Studio
def createStudioMatrix(movies):
    try:
        studio_details = list(map(str,(movies['Studio'])))
        studiolist = []
        for i in studio_details:
            split_studio = list(map(str, i.split(',')))
            for j in split_studio:
                if j not in studiolist:
                    studiolist.append(j)
        def binary(studio_list):
            binaryList = []
            
            for studio in studiolist:
                if studio in studio_list:
                    binaryList.append(1)
                else:
                    binaryList.append(0)
            
            return binaryList
        movies['studio_bin'] = movies['Studio'].astype('str').apply(lambda x: binary(x))
    except Exception as e:
        print('Exception in creating Studio Score Matrix ')
        print('Exception is ::',e)
        
def calculateStudioScore(movies):
    try:
        studioscores = []
        for i in range(0,movies.shape[0]):
            studioscore = calculatecore(i,'studio_bin')
            studioscores.append(studioscore)
        movies['studioscores'] = studioscores
    except Exception as e:
        print('Exception in calculating Studio Scores')
        print('Exception is ::',e) 
        

 
    
#Written By
def createWriterMatrix(movies):
    try:
        writer_details = list(map(str,(movies['Written By'])))
        writerlist = []
        for i in writer_details:
            split_writer = list(map(str, i.split(',')))
            for j in split_writer:
                if j not in writerlist:
                    writerlist.append(j)
        def binary(writer_list):
            binaryList = []
            
            for writer in writerlist:
                if writer in writer_list:
                    binaryList.append(1)
                else:
                    binaryList.append(0)
            
            return binaryList
        movies['writer_bin'] = movies['Written By'].astype('str').apply(lambda x: binary(x))
    except Exception as e:
            print('Exception in creating Writer Matrix')
            print('Exception is ::',e)
            
def calculateWriterScore(movies):
    try:
        writerscores = []
        for i in range(0,movies.shape[0]):
            writerscore = calculatecore(i,'writer_bin')
            writerscores.append(writerscore)
        movies['writerscores'] = writerscores
    except Exception as e:
            print('Exception in calculating Writer Scores')
            print('Exception is ::',e)  
            
#Rating            
def createRatingMatrix(movies):
    try:
        RatingList = ['PG','NR','R','PG-13']
        def binary(Rating_list):
            binaryList = []
            
            for Rating in RatingList:
                if Rating in Rating_list:
                    binaryList.append(1)
                else:
                    binaryList.append(0)
            
            return binaryList
        movies['Rating_bin'] = movies['Rating'].astype('str').apply(lambda x: binary(x))
    except Exception as e:
            print('Exception in creating Rating Matrix')
            print('Exception is ::',e)
            
def calculateRatingScore(movies):
    try:
        Ratingscores = []
        for i in range(0,movies.shape[0]):
            Ratingscore = calculatecore(i, 'Rating_bin')
            Ratingscores.append(Ratingscore)
        movies['Ratingscores'] = Ratingscores
    except Exception as e:
            print('Exception in calculating Rating Scores')
            print('Exception is ::',e)



#summary
def createSynopsisMatrix(movies):
    try:
        summary=movies['synopsis']
        tfidf = TfidfVectorizer(lowercase=True, norm=None, stop_words='english', use_idf=False)
        tfidf.fit_transform(summary).toarray()
        summarylist=tfidf.get_feature_names()
        def binary(summary_list):
            binaryList = []
            
            for summary in summarylist:
                if summary in summary_list:
                    binaryList.append(1)
                else:
                    binaryList.append(0)
            
            return binaryList
        movies['synopsis_bin'] = movies['synopsis'].astype('str').apply(lambda x: binary(x))        
    except Exception as e:
        print('Exception in creating Synopsis Matrix')
        print('Exception is ::',e)
        
def calculateSynopsisScore(movies):
    try:
        synopsisscores = []
        for i in range(0,movies.shape[0]):
            synopsisscore = calculatecore(i,'synopsis_bin')
            synopsisscores.append(synopsisscore)
        movies['summary_1scores'] = synopsisscores
    except Exception as e:
            print('Exception in calculating Synopsis Scores')
            print('Exception is ::',e)


def addSerailNos(movies):
    new_id=list(range(0,movies.shape[0]))
    movies['new_id']=new_id
# =============================================================================
# Cleaning Columns from dataFrame
print("Cleaning Started")
cleanAudienceScore(movies)
print("Completed : cleanAudienceScore")
cleanCriticScore(movies)
print("Completed : cleanCriticScore")
cleanRunTime(movies)
print("Completed : cleanRunTime")
cleanBoxOffice(movies)
print("Completed : cleanBoxOffice")


# =============================================================================

addSerailNos(movies)
print("Completed : adding Serial Nos")
# =============================================================================
# adding new Columns to dataFrame
calculateRatingDifference(movies)
print("Completed : calculateRatingDifference")

createGenreMatrix(movies)
print("Completed : createGenreMatrix")
calculateGenreScore(movies)
print("Completed : calculateGenreScore") 

createActorMatrix(movies)
print("Completed : createActorMatrix") 
calculateActorsScore(movies)
print("Completed : calculateActorsScore") 

createStudioMatrix(movies)
print("Completed : createStudioMatrix")   
calculateStudioScore(movies)
print("Completed : calculateStudioScore") 

createDirectorMatrix(movies)
print("Completed : createDirectorMatrix") 
calculateDirectorScore(movies)
print("Completed : calculateDirectorScore")   

createWriterMatrix(movies)
print("Completed : createWriterMatrix") 
calculateWriterScore(movies)
print("Completed : calculateWriterScore") 

createRatingMatrix(movies)
print("Completed : createRatingMatrix") 
calculateRatingScore(movies)
print("Completed : calculateRatingScore") 

createSynopsisMatrix(movies)
print("Completed : createSynopsisMatrix") 
calculateSynopsisScore(movies)
print("Completed : calculateSynopsisScore")  
# =============================================================================

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from nltk.corpus import stopwords
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
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
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.svm import SVR



# =============================================================================
# movies.fillna(method= 'ffill').astype(int)
# from sklearn.preprocessing import Imputer
# from sklearn.preprocessing import StandardScaler
# imp = Imputer(missing_values='NaN', strategy='median', axis=0)
# array = movies.values
# y = movies['RatingDiff'].values
# imp.fit(array)
# array_imp = imp.transform(array)
# 
# =============================================================================

# =============================================================================
# 
# 
# y2= y.reshape(1,-1)
# imp.fit(y2)
# y_imp= imp.transform(y2)
# 
# X = array_imp[:,0:4]
# Y = array_imp[:,4]
# 
# 
# 
# =============================================================================



# =============================================================================
# from sklearn import preprocessing
# lab_enc = preprocessing.LabelEncoder()
# X_train = lab_enc.fit_transform(X_train)
# 
# =============================================================================



movies = movies.drop('movie_id', 1)
train, test = train_test_split(movies, test_size=0.2)

X = train.values
y = test.values
seed = 7
scoring = 'accuracy'

validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, y, test_size=validation_size, random_state=seed)
# Spot Check Algorithms
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

