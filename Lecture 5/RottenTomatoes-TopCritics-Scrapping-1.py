# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 19:12:56 2018

@author: Harsh Kava
"""

# The program scraps the critic, rating, source,review date, review length from the input website link
# and write it to file called reviews.txt.
#

from bs4 import BeautifulSoup
import re
import time
import requests

def run(url):

    pageNum=2 # number of pages to collect

    fw=open('reviews.txt','w') # output file
	
    for p in range(1,pageNum+1): # for each page 

        print ('page',p)
        html=None

        if p==1: pageLink=url # url for page 1
        else: pageLink=url+'?page='+str(p)+'&sort=' # make the page url
		
        for i in range(5): # try 5 times
            try:
                #use the browser to access the url
                response=requests.get(pageLink,headers = { 'User-Agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36', })
                html=response.content # get the html
                break # we got the file, break the loop
            except Exception as e:# browser.open() threw an exception, the attempt to get the response failed
                print ('failed attempt',i)
                time.sleep(2) # wait 2 secs
				
		
        if not html:continue # couldnt get the page, ignore
        
        soup = BeautifulSoup(html.decode('ascii', 'ignore'),'lxml') # parse the html 

        reviews=soup.findAll('div', {'class':re.compile('review_table_row')}) # get all the review divs

        for review in reviews:            
            
            #fw.write(getCritic(review)+'\t'+getRating(review)+'\t'+getSource(review)+'\t'+getDate(review)+'\t'+getTextLen(review)+'\n') # write to file 
            #time.sleep(2)	# wait 2 secs 
            
            critic = getCritic(review)
            rating = getRating(review)
            source = getSource(review)
            reviewDate= getDate(review)
            textLen = getTextLen(review)
            fw.write('Critic Name :'+critic+' |Review Length :' +str(textLen)+ ' |Rating :'+rating+ ' |Source :'+source+ ' |Review Date:'+reviewDate+'\n' ) # write to file 
            print('Critic Name :'+critic+' |Review Length :' +str(textLen)+ ' |Rating :'+rating+ ' |Source :'+source+ ' |Review Date:'+reviewDate+'\n' ) # write to file 
            
    fw.close()
    


def getCritic(review):
    critic ='NA' # initialize critic 
    criticChunk=review.find('a',{'href':re.compile('/critic/')})
    if criticChunk: critic=criticChunk.text#.encode('ascii','ignore')
    
    return critic

def getRating(review):
    
    rating ='NA' # initialize rating
    
    ratingChunk = review.find('div',{'class':'review_icon icon small rotten'})
    if(ratingChunk):
        rating ='rotten'
        return rating
    else:
        ratingChunk = review.find('div',{'class':'review_icon icon small fresh'})
        if(ratingChunk): rating = 'fresh' 
        return rating
        
def getSource(review):
    source = 'NA'
    sourceChunk = review.find('a',{'href':re.compile('/source-')})
    if(sourceChunk): source = sourceChunk.text.strip()
    return source
    

def getDate(review):
    reviewDate = 'NA'
    dateChunk = review.find('div',{'class':'review_date subtle small'})
    if(dateChunk): reviewDate = dateChunk.text.strip()
    return reviewDate
    
    
def getTextLen(review):
    text='NA' # initialize text
    textLength = 'NA'
    textChunk=review.find('div',{'class':'the_review'})
    if textChunk: text=textChunk.text#.encode('ascii','ignore')
    if text: textLength =len(text.strip())
    
    return textLength
    


if __name__=='__main__':
    url='https://www.rottentomatoes.com/m/space_jam/reviews/?type=top_critics'
    run(url)
