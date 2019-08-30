# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 19:19:08 2018

@author: Harsh Kava
"""

import shutil
import os

directory="D:/Courses/04_Web Analytics/Study Material/Lecture 8/20_newsgroups/"
print('Current working directory path:',os.getcwd())

os.chdir('D:/Courses/04_Web Analytics/Study Material/Lecture 8/20_newsgroups/')
for file in os.listdir(directory):
    print(file)
    if file == "comp.windows.x":
        print('comp.windows.x found')
        os.rename("comp.windows.x","comp")
    elif file == "rec.sport.baseball":
        os.rename("rec.sport.baseball","sports")            
    elif file == "talk.politics.misc":
        os.rename("talk.politics.misc","politics")
    elif file == "rec.autos":
        os.rename("rec.autos","rec")  
    else:
        continue
   
comp_test="D:/Courses/04_Web Analytics/Study Material/Lecture 8/20_newsgroups/train_comp"
sports_test="D:/Courses/04_Web Analytics/Study Material/Lecture 8/20_newsgroups/train_sports"
politics_test="D:/Courses/04_Web Analytics/Study Material/Lecture 8/20_newsgroups/train_politics"
rec_test="D:/Courses/04_Web Analytics/Study Material/Lecture 8/20_newsgroups/train_rec"

for fil in os.listdir(directory):
    if (fil.startswith("comp.")):
        shutil.move(fil,comp_test)
    elif ("sport." in fil):
        shutil.move(fil,sports_test)
    elif (fil.startswith("rec.")):
        shutil.move(fil,rec_test)
    elif ("politics." in fil):
        shutil.move(fil,politics_test)
    else:
        continue
    
def move_file(folder):
    RootDir1 = r'D:/Courses/04_Web Analytics/Study Material/Lecture 8/20_newsgroups'
    TargetFolder = r'D:/Courses/04_Web Analytics/Study Material/Lecture 8/20_newsgroups/comp_test'
    for root, dirs, files in os.walk((os.path.normpath(RootDir1)), topdown=False):
            for fil in files:
                if (fil.startswith("comp.")):
                    SourceFolder = os.path.join(root,fil)
                    print (SourceFolder)
                    shutil.move(SourceFolder,TargetFolder)
                elif ("sport." in fil):
                    shutil.move(fil,sports_test)
                elif (fil.startswith("rec.")):
                    shutil.move(fil,rec_test)
                elif ("politics." in fil):
                    shutil.move(fil,politics_test)
                else:
                    continue                
               
RootDir1 = r'D:/Courses/04_Web Analytics/Study Material/Lecture 8/20_newsgroups'
TargetFolder = r'D:/Courses/04_Web Analytics/Study Material/Lecture 8/20_newsgroups/comp_test'
for root, dirs, files in os.walk((os.path.normpath(RootDir1)), topdown=False):
        for fil in files:
            print (fil)
            if ("comp." in fil):
                SourceFolder = os.path.join(root,fil)
                print (SourceFolder)
                shutil.move(SourceFolder,TargetFolder)
                    
                    
import glob



read_files = glob.glob("D:/Courses/04_Web Analytics/Study Material/Lecture 8/20_newsgroups/train_rec/A/*")
with open("train_rec_all.txt", "w") as outfile:
    for f in read_files:
        with open(f, "r") as infile:
            line15=infile.readlines()[15:]
            outfile.writelines(line15)
            
            
def file_create(strr):
    read_files = glob.glob("D:/Courses/04_Web Analytics/Study Material/Lecture 8/20_newsgroups/"+strr+"/*")
    with open(strr+".txt", "w") as outfile:
        for f in read_files:
            with open(f, "r") as infile:
                line15=infile.readlines()[15:]
                outfile.writelines(line15)


file_create("train_rec")
file_create("train_poli")
file_create("train_sports")
file_create("train_comp")    
    


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

def loaddata(fname):
    reviews=[]
    f=open(fname)
    for line in f:
        reviews.append(line.lower())
    f.close()
    return reviews


train_comp,label_comp=loaddata('train_comp.txt'),'comp'
train_poli,label_poli=loaddata('train_poli.txt'),'poli'
train_rec,label_rec=loaddata('train_rec.txt'),'rec'
train_sports,label_sports=loaddata('train_sports.txt'),'sports'

def loadfolder1(folder):
    read_files = glob.glob("D:/Courses/04_Web Analytics/Study Material/Lecture 8/20_newsgroups/"+folder+"/*")
    rev=[]
    lab=[]
    for f in read_files:
        fil=open(f)
        ff=fil.readlines()[15:]
        #rev.append(ff)
        #fill=file.readlines()[15:]
        for line in ff:
            if "_" in folder:
                lab.append(folder[6:])
            else:
                lab.append(folder)
            ll=line.strip()
            rev.append(ll.lower())
        #rev.append(ff.lower())
        fil.close()
    return rev,lab
    
def loadfolder(folder):
    read_files = glob.glob("D:/Courses/04_Web Analytics/Study Material/Lecture 8/20_newsgroups/"+folder+"/*")
    rev=[]
    for f in read_files:
        fil=open(f)
        str1=''
        ff=fil.read().splitlines()[15:]
        str1 = ''.join(ff)        
        rev.append(str1)
        fil.close()
    return rev
    
train_comp=loadfolder('train_comp')
#str1 = ''.join(map(str,train_comp))
train_pol,label_poli=loadfolder('train_politics')
train_rec,label_rec=loadfolder('train_rec')
train_sports,label_sports=loadfolder('train_sports')

test_comp=loadfolder('comp')
test_poli,label_polit=loadfolder('politics')
test_rec,label_rect=loadfolder('rec')
test_sports,label_sportst=loadfolder('sports')

(label_comp,label_poli,label_rec,label_sports,label_compt,label_polit,label_rect,label_sportst)=('comp','politics','rec','sports','comp','politics','rec','sports')

test_comp,label_compt='D:/Courses/04_Web Analytics/Study Material/Lecture 8/20_newsgroups/comp/*','comp'
test_poli,label_polit='D:/Courses/04_Web Analytics/Study Material/Lecture 8/20_newsgroups/politics/*','poli'
test_rec,label_rect='D:/Courses/04_Web Analytics/Study Material/Lecture 8/20_newsgroups/rec/*','comp'
test_sports,label_sportst='D:/Courses/04_Web Analytics/Study Material/Lecture 8/20_newsgroups/sports/*','sports'


import numpy as np
print (np.shape(train_comp))

counter=CountVectorizer(analyzer='word')
#vect=map(counter.fit,train_comp)
counter.fit(train_comp)
counter.fit(train_poli)
counter.fit(train_rec)
counter.fit(train_sports)

counts_train_comp=map(counter.transform,train_comp)

counts_train_comp=counter.transform(train_comp)
counts_train_poli=counter.transform(train_poli)
counts_train_rec=counter.transform(train_rec)
counts_train_sports=counter.transform(train_sports)

counts_test_comp=map(counter.transform,test_comp)

counts_test_comp=counter.transform(test_comp)
counts_test_poli=counter.transform(test_poli)
counts_test_rec=counter.transform(test_rec)
counts_test_sports=counter.transform(test_sports)

clf=MultinomialNB()

clf.fit(counts_train_comp,label_comp)
clf.fit(counts_train_poli,label_poli)
clf.fit(counts_train_rec,label_rec)
clf.fit(counts_train_sports,label_sports)

pred1=clf.predict(counts_test_comp)
pred2=clf.predict(counts_test_poli)
pred3=clf.predict(counts_test_rec)
pred4=clf.predict(counts_test_sports)

print (accuracy_score(pred1,label_compt))
print (accuracy_score(pred2,label_polit))
print (accuracy_score(pred3,label_rect))
print (accuracy_score(pred4,label_sportst))    
    
    
    
    
      