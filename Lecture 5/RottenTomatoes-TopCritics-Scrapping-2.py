# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 20:34:45 2018

@author: Harsh Kava
"""

"""
Input: movie id in rottentomatoes
Output: all reviews in "top critics" as a list of tuples (reviewer, date, description, score)

"""
from bs4 import BeautifulSoup
import re
import time
import requests

def getReviews(movie_id):
   
    reviews=[]  # variable to hold all reviews
   
    page_url="https://www.rottentomatoes.com/m/"+movie_id+"/reviews/?type=top_critics"
    print(page_url)
    page = requests.get(page_url)
   
    if page.status_code==200:   
       
        # insert your code to process page content
        reviews = run(page_url)
        
    return reviews


def run(url):
    
    reviewsList=[]
    
    html=None
	
    for i in range(5): # try 5 times
        try:
            #use the browser to access the url
            response=requests.get(url,headers = { 'User-Agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36', })
            html=response.content # get the html
            break # we got the file, break the loop
        except Exception as e:# browser.open() threw an exception, the attempt to get the response failed
            print ('failed attempt',i)
            time.sleep(2) # wait 2 secs
				
		
    #if not html:continue # couldnt get the page, ignore
    
    soup = BeautifulSoup(html.decode('ascii', 'ignore'),'lxml') # parse the html 

    reviews=soup.findAll('div', {'class':re.compile('review_table_row')}) # get all the review divs

    for review in reviews:            
        
        
        critic = getCritic(review)
        reviewDate= getDate(review)
        text = getText(review)
        score = getScore(review)
        
        x = 'Critic Name:'+critic+' |Review Date:'+reviewDate+' |Review:'+text+ ' |Score:'+score 
        print(x)
        reviewsList.append(x) 
        
        
    return reviewsList
            
    #fw.close()
    
def getCritic(review):
    critic ='NA' # initialize critic 
    criticChunk=review.find('a',{'href':re.compile('/critic/')})
    if criticChunk: critic=criticChunk.text#.encode('ascii','ignore')
    
    return critic        
    
def getScore(review):
    #score = 'NA'
    scoreChunk = review.find('div',{'class':'small subtle'})
    if(scoreChunk): score = scoreChunk.text.strip()
    try:
        score = score.split(':')[1]
    except:
        score='NA'    
    return score
    

def getDate(review):
    reviewDate = 'NA'
    dateChunk = review.find('div',{'class':'review_date subtle small'})
    if(dateChunk): reviewDate = dateChunk.text.strip()
    return reviewDate
    
    
def getText(review):
    text='NA' # initialize text
    textChunk=review.find('div',{'class':'the_review'})
    if textChunk: text=textChunk.text#.encode('ascii','ignore')
    return text
    
if __name__ == "__main__": 
   
    #mpg_plot()
   
    movie_id='finding_dory'
    reviews=getReviews(movie_id)
    #print(reviews)
    for i in reviews:
        print(i+'\n')
        