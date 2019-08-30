# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 16:11:49 2018

@author: Harsh Kava
"""

f = open('Team_3_Data.txt','r',encoding="utf8")
my_file = open('Our_data.txt','w')

lines = f.readlines()[1:]

for i in lines:
    print(i)
    try:
        
        mov = i.split('\t')
        my_file.write(mov[0]+'\t'+mov[2]+'\t'+mov[1]+'\t'+mov[19]
        +'\t'+ 'NA' +'\t'+ 'NA'+'\t'+mov[14]+'\t'+mov[11]+'\t'+
        mov[18]+'\t'+mov[12]+'\t'+mov[16]+'\t'+mov[17]+'\t'+
        mov[10]+'\t'+mov[13]+'\n')

    except Exception as e:
        print('Exception in parsing the movie and creating the dictionary for :',i)
        print('Exception is ::',e)


my_file.close()
f.close()