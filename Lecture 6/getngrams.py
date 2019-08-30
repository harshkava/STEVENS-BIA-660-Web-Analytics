# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 19:15:38 2018

@author: Harsh Kava
"""

"""
processSentence(sentence,posLex,negLex,tagger):  The parameters of this function are a sentence (a string), a set positive words, a set of negative words, and a POS tagger.  The function should return a list with all the 4-grams in the sentence that have the following structure:                                                   

not <any word> <pos/neg word> <noun>

For example: not a good idea

"""


"""

The script includes the following pre-processing steps for text:
- Sentence Splitting
- Term Tokenization
- Ngrams
- POS tagging

The processSentence function includes all 4grams of the form: not <any word> <pos/neg word> <noun>

POS tags list: https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
"""

import nltk
from nltk.util import ngrams
import re
from nltk.tokenize import sent_tokenize
from nltk import load
import operator

D={}

def getAdvAdjTwograms(terms,posLex,negLex,nouns):
  
    result=[]
    fourgrams = ngrams(terms,4) #compute 4-grams    
   	 #for each 4gram
    for tg in fourgrams: 
        if tg[0].lower() == 'not':
            if tg[2] in posLex or tg[2] in negLex:
                if tg[3] in nouns: # if the 2gram is a an adverb followed by an adjective
                    result.append(tg)

    return result  

# return all the terms that belong to a specific POS type
def getPOSterms(terms,POStags,tagger):
	
    tagged_terms=tagger.tag(terms)#do POS tagging on the tokenized sentence
    POSterms=[]
    for pair in tagged_terms:
        if pair[1].startswith(POStags):
            POSterms.append(pair[0])
    return POSterms

#function that loads a lexicon of positive words to a set and returns the set
def loadLexicon(fname):
    newLex=set()
    lex_conn=open(fname)
    #add every word in the file to the set
    for line in lex_conn:
        newLex.add(line.strip())# remember to strip to remove the lin-change character
    lex_conn.close()

    return newLex
 
def processSentence(fpath):

    #make a new tagger
    _POS_TAGGER = 'taggers/maxent_treebank_pos_tagger/english.pickle'
    tagger = load(_POS_TAGGER)
    
    #creating Positive and Negative lexiconss 
    posLex=loadLexicon('positive-words.txt')
    negLex=loadLexicon('negative-words.txt')
    
    #read the input
    f=open(fpath)
    text=f.read().strip()
    f.close()

    #split sentences
    sentences=sent_tokenize(text)
    #print ('NUMBER OF SENTENCES: ',len(sentences))
    
    adjAfterAdv=[]

    # for each sentence
    for sentence in sentences:
       #print("enetering processSentence")
       sentence=re.sub('[^a-zA-Z\d]',' ',sentence)#replace chars that are not letters or numbers with a spac
       sentence=re.sub(' +',' ',sentence).strip()#remove duplicate spaces

        #tokenize the sentence
       terms = nltk.word_tokenize(sentence.lower()) 
        #print("terms",terms)
    
       POStags='NN' # POS tags of interest i.e. Noun		
       nouns=getPOSterms(terms,POStags,tagger)  #gets all nouns from sentence
       #print(nouns)
        
        #get the results for this sentence 
       adjAfterAdv+=getAdvAdjTwograms(terms, posLex,negLex,nouns)
       
       #filling dictionary D with word count
       for word in terms:
           if word in D:
               D[word] = D[word] +1
           else:
               D[word] = 1
 
    return adjAfterAdv

#getTop3(D): The only parameter of this function is a dictionary D.  All the values in the dictionary are integers. The function returns a list of the keys with the 3 largest values in the dictionary.
def getTop3(D):
    result = dict(sorted(D.items(), key=operator.itemgetter(1), reverse=True)[:3])
    return list(result.keys())

if __name__=='__main__':
    print (processSentence('input.txt'))
    print('3 largest values in the dictionary ::',getTop3(D))



