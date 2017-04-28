# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 13:00:42 2017

@author: monarang
"""

from bs4 import BeautifulSoup
from PIL import Image
import requests
from StringIO import StringIO
import pickle

if not os.path.isfile('outfile'):
    href = "http://www.delhischooladmissions.in/search_product.php?searchtxt=&search2=Go"
    school_urls = []
    s = requests.Session()
    first_page = s.get(href)
    
    soup = BeautifulSoup(first_page.content)
    
    for i in soup.findAll('a', href=True, text='\n\t\t\t\t\t\t\t\t        View Details'):
        school_urls.append(i.get('href'))
        
    for i in range(2,99):
        new_href = href + "&page="+str(i)
        first_page = s.get(new_href)
        soup = BeautifulSoup(first_page.content)
        for i in soup.findAll('a', href=True, text='\n\t\t\t\t\t\t\t\t        View Details'):
            school_urls.append(i.get('href'))
        print("Page "+str(i)+ " is done")
      
    
    with open('outfile', 'wb') as fp:
        pickle.dump(school_urls, fp)


with open ('outfile', 'rb') as fp:
    saved_school_urls = pickle.load(fp)
 
detail_text = []    

for page_num,j in enumerate(saved_school_urls):
    url = "http://www.delhischooladmissions.in"+j
    s1 = requests.Session()
    each_page = s1.get(url)
    
    soup = BeautifulSoup(each_page.content)

    NormalText=soup.find("td",class_="NormalText").text
    detail_text.append(NormalText)
    print "Page Number " + str(page_num) +" is completed"
    
with open('detail_text', 'wb') as fp:
    pickle.dump(detail_text, fp)
   
with open ('detail_text', 'rb') as fp:
    saved_detail_text = pickle.load(fp)
 
import numpy as np
np.savetxt("saved_detail_text.csv", saved_detail_text, delimiter=",", fmt='%s')

import csv

with open("saved_detail_text.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(saved_detail_text)

with open("output.csv",'wb') as resultFile:
    wr = csv.writer(resultFile, dialect='excel')
    wr.writerow([unicode(s).encode("utf-8") for s in saved_detail_text])
    
    