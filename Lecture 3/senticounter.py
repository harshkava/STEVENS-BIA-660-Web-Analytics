# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 18:51:48 2018

@author: Harsh Kava
"""

"""
The function :

- Accepts a parameter the path to a text file. The text file has one review per line. 

- Read the list of positive words from the positive-words.txt file.

- Creates a dictionary that includes one key for each positive word that appears in the input text file.
  The dictionary maps each of these positive words to the number of reviews that include it.
  For example, if the word "great" appears in 5 reviews, then the dictionary maps the key "great" to the value 5. 

- Returns the dictionary 

"""

def loadPositiveWords(fname):
    wordList=set()
    lex_conn=open(fname)
    
    for line in lex_conn:
        wordList.add(line.strip())# remember to strip to remove the line -change character
    lex_conn.close()

    return wordList

def run(path):
    
    #load the positive lexicon
    positive_words = loadPositiveWords("positive-words.txt")
    
    fileIn = open(path)
    
    mydict = {}
    
    for line in fileIn:  # for every line in the file (1 review per line)
        
        line = line.lower().strip()
        #print(line)
        words = set(line.split(" "))# slit on the space to get list of words

        for word in words:
            if(word in positive_words):
                if(word in mydict.keys()):
                    count = mydict[word]+1
                    mydict[word] = count
                else:
                    mydict[word] = 1
    fileIn.close()
    return mydict
                
if __name__ == "__main__": 
        mydict=run('textfile')
        print(mydict)
        
        
        