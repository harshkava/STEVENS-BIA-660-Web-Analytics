from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from fake_useragent import UserAgent
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import re,time,os,codecs

code='abc' # THIS IS THE VALUE THAT YOU SHOULD CHANGE

chain='tgi'

#make browser
ua=UserAgent()
dcap = dict(DesiredCapabilities.PHANTOMJS)
dcap["phantomjs.page.settings.userAgent"] = (ua.random)
service_args=['--ssl-protocol=any','--ignore-ssl-errors=true']
driver = webdriver.Chrome('chromedriver.exe',desired_capabilities=dcap,service_args=service_args)

if not os.path.exists('users/'+chain):os.mkdir('users/'+chain)

f=open(chain+'_users')

done=set()

if os.path.exists('done_usrs.txt'): 
    with open('done_usrs.txt') as f: done.update([x.strip() for x in f])


dw=open('done_usrs.txt','a')

seen=set()

f=open(chain+'_users')
#for each user
for line in f:
    ignore=False
    user=line.strip()
    if not re.search(user[0].lower(),code):continue
    
    if user in seen:continue
    seen.add(user)
    
    if user in done:continue
    
    #make the link
    link='https://www.tripadvisor.com/members/'+user
    
    #visit the link
    driver.get(link)
    time.sleep(2)
    #find the "Restaurants" Button and scroll to it    
    
    found=False
    for nn in['3','2']:
        time.sleep(2)
        try:restaurantsButton = WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.XPATH,'//*[@id="BODYCON"]/div[3]/div[2]/ul/li['+nn+']')))
        except: continue
        if 'Restaurants' in restaurantsButton.text: 
            found=True
            break
    
    if not found:
        print ('MISSING',user)
        done.add(user)
        dw.write(user+'\n')
        dw.flush()
        continue
    

    print('Location',restaurantsButton.location)
    
    loc=restaurantsButton.location['y']-100
    print('Loc',loc)
    
    driver.execute_script("window.scrollTo(0, "+str(loc)+");")
    
    #record the number of restaurant reviews that this user has
    number=int(re.search('[\d,]+',restaurantsButton.text).group(0).replace(',',''))
    #print ('RT',restaurantsButton.text)
  
    #click on the restaurant button
    try:restaurantsButton.click()
    except:
        print ('missing RESTAURANT button')
        done.add(user)
        continue
    lw=open('users/'+chain+'/profile_links.txt','a')
    
    
    page_htmls=[]
    rlinks=set()
    while True: # go over all the pages of restaurant reviews
        time.sleep(0.5)
        #print (page)    
        try:html=driver.page_source# get the html
        except: 
            #print ('bad inspector')
            #done.add(user)
            break
                
        page_htmls.append(html)
        links_in_this_page=set()
        try:nextB= WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.XPATH,'//*[@id="cs-paginate-next"]')))# find the "Next" button
        except:
            ignore=True
            break
        
        Ms=re.finditer('feedRestaurant.+?(/ShowUserReviews.+?html)',html) # find all the restaurant reviews in the html
        for M in Ms: links_in_this_page.add(M.group(1))
        
        rlinks.update(links_in_this_page)# update the global set
        
        
        # scroll to the Next button
        try:
            loc=nextB.location['y']-100
            driver.execute_script("window.scrollTo(0, "+str(loc)+");")
        
            #wait until its clickable and click it
            nextB=WebDriverWait(driver, 3).until(EC.element_to_be_clickable((By.XPATH,'//*[@id="cs-paginate-next"]')))
        
            if 'disabled' in nextB.get_attribute('class'):break # Next button is disabled, we are in the last page
            nextB.click()    
        except:
            ignore=True
            break
            
        time.sleep(1)
      
    
    if ignore:
        print ('ignoring',user)
        continue
    missing_perc=abs(number-len(rlinks))/float(number)
    if missing_perc<0.1:
        dw.write(user+'\n')
        dw.flush()
        print(user)
        done.add(user)
        if not os.path.exists('users/'+chain+'/'+user):os.mkdir('users/'+chain+'/'+user)
        page=1
        for html in page_htmls:
            with codecs.open('users/'+chain+'/'+user+'/'+str(page), 'w',encoding='utf8') as fw: fw.write(html)# write the th
            page+=1
        
    else:
        print (user,'total',number,len(rlinks),missing_perc)
    
    for rlink in rlinks:
        lw.write(user+'\t'+rlink+'\n')
    
lw.close() 
dw.close()   