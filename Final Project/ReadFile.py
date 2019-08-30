# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 17:23:15 2018

@author: Harsh Kava
"""
import re
# =============================================================================
# movie_id = 'NA'
# audience_score = 'NA'
# critic_score = 'NA'
# actor_names = 'NA'
# actor_links = 'NA'
# synopsis = 'NA'
# in_theaters = 'NA'
# genre = 'NA'
# studio = 'NA'
# directed_by = 'NA'
# runtime = 'NA'
# box_office = 'NA'
# rating = 'NA'
# written_by = 'NA'
# =============================================================================

#Creating a List for each Movie
movies = []

f = open('Our_data.txt','r')

lines = f.readlines()[1:]

for i in lines:
    #print(i)
    try:
        mov_details = i.split('\t')
        #print(mov_details)
        mov = {}        #creating a dictionary
        mov['movie_id'] = mov_details[0]
        mov['audience_score'] = mov_details[1]
        mov['critic_score'] = mov_details[2]
        mov['actor_names'] = mov_details[3]
        mov['actor_links'] = mov_details[4]
        mov['synopsis'] = mov_details[5]
        mov['in_theaters'] = (mov_details[6]).replace(u'\xa0', u' ')
        mov['genre'] = mov_details[7]
        mov['studio'] = mov_details[8]
        mov['directed_by'] = mov_details[9]
        mov['runtime'] = mov_details[10]
        mov['box_office'] = mov_details[11]
        mov['rating'] = mov_details[12]
        mov['written_by'] = mov_details[13]
        
        movies.append(mov)
    except Exception as e:
        print('Exception in parsing the movie and creating the dictionary for :',i)
        print('Exception is ::',e)
    
f.close()