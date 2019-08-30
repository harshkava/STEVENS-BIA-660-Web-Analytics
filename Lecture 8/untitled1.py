# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 19:32:13 2018

@author: Harsh Kava
"""


    reviews=[]
    labels=[]
    f=open(fname)
    for line in f:
        review,rating=line.strip().split('\t')  
        reviews.append(review.lower())    
        labels.append(rating)
    f.close()
    return reviews,labels

rev_train,labels_train=loadData('reviews_train.txt')
rev_test,labels_test=loadData('reviews_test.txt')