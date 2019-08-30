

"""
Created on Wed Apr 18 17:01:41 2018

@author: Harsh Kava
"""

from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from fake_useragent import UserAgent
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.common.by import By
import time

ua=UserAgent()
dcap = dict(DesiredCapabilities.PHANTOMJS)
dcap["phantomjs.page.settings.userAgent"] = (ua.random)
service_args=['--ssl-protocol=any','--ignore-ssl-errors=true']
driver = webdriver.Chrome('chromedriver.exe',desired_capabilities=dcap,service_args=service_args)


def login(username,password):
    #access website
    driver.get('https://www.facebook.com/')
    
    #Accessing Login frame
    form=driver.find_element_by_id('login_form')
    form.click()
    
    #Entering email details
    email = form.find_element_by_id('email')
    email.send_keys(username)
    #time.sleep(1)
    
    #Entering password details
    pwd = form.find_element_by_id('pass')
    pwd.send_keys(password)
    time.sleep(1)
    
    #Clicking the login button
    button=WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.ID, 'loginbutton')))
    button.click()

def scrapData():
    time.sleep(1)
    driver.get("https://www.facebook.com/pg/ATT/community/?ref=page_internal")
    time.sleep(3)
    
    fw=open('Att_reviews.txt','w')
    
    try:
        postList = driver.find_element_by_id('u_8_11')
        
        for post in postList:
            print(post)
            print(post.text)
        
    except Exception as e:
        print(e)

    
login("######","######")   # use your email address and password to login
scrapData()