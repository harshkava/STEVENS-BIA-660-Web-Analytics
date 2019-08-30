"""
A simple script that demonstrates how we classify textual data with sklearn.

"""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
import math

#read the reviews and their polarities from a given file
def loadData(fname):
    reviews=[]
    labels=[]
    f=open(fname)
    for line in f:
        review,rating=line.strip().split('\t')  
        reviews.append(review.lower())    
        labels.append(int(rating))
    f.close()
    return reviews,labels

rev_train,labels_train=loadData('reviews_train.txt')
rev_test,labels_test=loadData('reviews_test.txt')


#Build a counter based on the training dataset
counter = CountVectorizer()
counter.fit(rev_train)


#count the number of times each term appears in a document and transform each doc into a count vector
counts_train = counter.transform(rev_train)#transform the training data
counts_test = counter.transform(rev_test)#transform the testing data

#print(counts_train)
#train classifier
#clf = MultinomialNB()
#clf = ExtraTreesClassifier()
#clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=10)
#clf =MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto')
#clf = GaussianNB()
#clf = BernoulliNB()
#clf = LogisticRegression(random_state=100,solver='liblinear', max_iter=100, multi_class='ovr')
#clf = ExtraTreesClassifier(n_estimators=10000, max_depth=None, min_samples_split=2)
clf = RandomForestClassifier(n_estimators=2500, n_jobs=15,criterion="entropy",max_features='log2',random_state=150,max_depth=600,min_samples_split=163)
#clf = AdaBoostClassifier(n_estimators=1000
#clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=None, random_state=2)
#clf = RandomForestClassifier(random_state=10)

#clf = GradientBoostingClassifier(n_estimators=1000, learning_rate=1.0    max_depth=1, random_state=0).fit(counts_train, labels_train)

#train all classifier on the same datasets

clf.fit(counts_train,labels_train)

#use hard voting to predict (majority voting)
pred=clf.predict(counts_test)

#print accuracy
print (accuracy_score(pred,labels_test))



