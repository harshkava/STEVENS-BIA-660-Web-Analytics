# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 18:31:43 2018

@author: shrey
"""


import pandas as pd
import csv


movies = pd.read_csv("sample.txt",sep='\t',header=0)
actorFBList = pd.read_csv("actor_rating.csv")
#Actor_FB_Likes = list(map(str,(actor_likes['actor_1_facebook_likes'])))
actorFBList['actor_1_name'] 
# =============================================================================
# 
actorFBList = csv.reader(open("actor_rating.csv", 'r',encoding="utf8"))
actorFBList
actor_likes = {}
for row in actorFBList:
     print(row)
     k, v = row
     actor_likes[k] = v
# 
# =============================================================================
#Actor



import csv  

actor_likes = {}

with open('actor_rating.csv',encoding="utf8") as csvfile:  
    reader = csv.DictReader(csvfile)
    for row in reader:
        #print(row['actor_1_name'])
        print(row)
        #k, v = row
        actor_likes[row['actor_1_name']] = row['actor_1_facebook_likes']
        
actor_likes 
        
        
actor_list_per_movie = list(map(str,(movies['actor_names'])))
actor_list_per_movie


Actorlist = []


for i in actor_list_per_movie:
    split_actor = list(map(str, i.split(',')))
    for j in split_actor:
        if j not in Actorlist:
            Actorlist.append(j)

Actorlist


def binary(Actorlist):
    binaryList = []
    
    for Actor in Actorlist:

        if Actor in actor_likes:
            binaryList.append(actor_likes[Actor])
        else:
            binaryList.append(0)
    
    return binaryList

print(binary(Actorlist))


