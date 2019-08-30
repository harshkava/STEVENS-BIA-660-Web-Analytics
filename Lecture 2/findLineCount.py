# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 19:26:59 2018

@author: Harsh Kava
The script defines a function run(). The function accepts as input the path to a text file and 2 words. It then returns the number of times that each
word appears in the file.
"""

#define a new function
def findWordCount(path,word1,word2):
    
    dict ={}  # new dictionary. Maps each word to each frequency

    # initialize the frequency of the two words to zero.
    dict[word1] =0
    dict[word2] =0
    
    file = open(path)  # open a connection to the file
    
    for i in file:
        # lower() converts all the letters in the string to lower-case
        # strip() removes blank space from the start and end of the string
        # split(c) splits the string on the character c and returns a list of the pieces. For example, "A1B1C1D".split('1')" returns [A,B,C,D]

        bool1 = False
        bool2 = False
        
        words = i.lower().strip().split(' ')
        #print(words)

        for word in words:
            if(word == word1):
                bool1 = True
            elif(word == word2):
                bool2 = True
        
        if(bool1 == True): dict[word1] +=1
        if(bool2 == True): dict[word2] +=1
    
        
    file.close() #close the connection to the text file

    return dict[word1],dict[word2]

# use the function
print(findWordCount('input_textfile','blue','yellow'))
print(findWordCount('input_textfile','name','kate'))