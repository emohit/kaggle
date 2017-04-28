# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 18:58:53 2016

@author: monarang
"""

from bs4 import BeautifulSoup
from PIL import Image
import requests
from StringIO import StringIO

s = requests.Session()
first_page = s.get("http://www.cbseaff.nic.in/cbse_aff/schdir_Report/userview.aspx")
first_page_soup = BeautifulSoup(first_page.content)

view_state = first_page_soup.find(id='__VIEWSTATE')['value']
event_validator = first_page_soup.find(id='__EVENTVALIDATION')['value']
hed ={
'Accept' : 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
'Accept-Encoding' : 'gzip, deflate',
'Accept-Language' : 'en-US,en;q=0.8,hi;q=0.6',
'Cache-Control' : 'max-age=0',
'Connection' : 'keep-alive',
'Content-Length' : '658',
'Content-Type' : 'application/x-www-form-urlencoded',
'Cookie' : first_page.headers['Set-Cookie'].split(';')[0],
'Host' : 'cbseaff.nic.in',
'Origin' : 'http://cbseaff.nic.in',
'Referer' : 'http://cbseaff.nic.in/cbse_aff/schdir_Report/userview.aspx',
'Upgrade-Insecure-Requests' : '1',
'User-Agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36'
}

payload = {
'__EVENTTARGET' : 'optlist$0',
'__EVENTARGUMENT' : '',
'__LASTFOCUS' : '',
'__VIEWSTATE' : view_state,
'__VIEWSTATEENCRYPTED' : '',
'__EVENTVALIDATION' : event_validator,
'optlist' : 'Search By KeyWord',
'keytext' : ''
}

keyword_page = s.request('POST','http://cbseaff.nic.in/cbse_aff/schdir_Report/userview.aspx',data = payload,headers=hed)
keyword_page_soup = BeautifulSoup(keyword_page.content)

view_state_key = keyword_page_soup.find(id='__VIEWSTATE')['value']
event_validator_key = keyword_page_soup.find(id='__EVENTVALIDATION')['value']

hed1= {
'Accept' : 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
'Accept-Encoding' : 'gzip, deflate',
'Accept-Language' : 'en-US,en;q=0.8,hi;q=0.6',
'Cache-Control' : 'max-age=0',
'Connection' : 'keep-alive',
'Content-Length' : '670',
'Content-Type' : 'application/x-www-form-urlencoded',
'Cookie' : first_page.headers['Set-Cookie'].split(';')[0],
'Host' : 'cbseaff.nic.in',
'Origin' : 'http://cbseaff.nic.in',
'Referer' : 'http://cbseaff.nic.in/cbse_aff/schdir_Report/userview.aspx',
'Upgrade-Insecure-Requests' : '1',
'User-Agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36'
}

payload1 = {
'__EVENTTARGET' : '',
'__EVENTARGUMENT' : '',
'__LASTFOCUS' : '',
'__VIEWSTATE' : view_state_key,
'__VIEWSTATEENCRYPTED' : '',
'__EVENTVALIDATION' : event_validator_key,
'optlist' : 'Search By KeyWord',
'keytext' : 'a',
'search' : 'Search'
}

school_page = s.request('POST','http://cbseaff.nic.in/cbse_aff/schdir_Report/userview.aspx',data = payload1,headers=hed1)
school_page_soup = BeautifulSoup(school_page.content)

table_t1 = school_page_soup.find('table',id='T1')
table_with_cellpadding_1 = table_t1.find_all('table',cellpadding=1)
#td_center= table_t1.find_all('td', {'align': 'left'})

import re

q=[]
for e,i in enumerate(table_with_cellpadding_1):
    l=[]
    for j in i.find_all('td', {'align': 'left'}):
        l.append(re.sub('[\n\t\r]','',j.text))
    q.append(l)

#18655 number of entries

for z in range(1,747):
    view_state_key = school_page_soup.find(id='__VIEWSTATE')['value']
    event_validator_key = school_page_soup.find(id='__EVENTVALIDATION')['value']
    
    hed1= {
    'Accept' : 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Encoding' : 'gzip, deflate',
    'Accept-Language' : 'en-US,en;q=0.8,hi;q=0.6',
    'Cache-Control' : 'max-age=0',
    'Connection' : 'keep-alive',
    'Content-Length' : '670',
    'Content-Type' : 'application/x-www-form-urlencoded',
    'Cookie' : first_page.headers['Set-Cookie'].split(';')[0],
    'Host' : 'cbseaff.nic.in',
    'Origin' : 'http://cbseaff.nic.in',
    'Referer' : 'http://cbseaff.nic.in/cbse_aff/schdir_Report/userview.aspx',
    'Upgrade-Insecure-Requests' : '1',
    'User-Agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36'
    }
    
    payload1 = {
    '__EVENTTARGET' : '',
    '__EVENTARGUMENT' : '',
    '__LASTFOCUS' : '',
    '__VIEWSTATE' : view_state_key,
    '__VIEWSTATEENCRYPTED' : '',
    '__EVENTVALIDATION' : event_validator_key,
    'optlist' : 'Search By KeyWord',
    'keytext' : 'a',
    'Button1' : 'Next >>'
    }
    
    school_page = s.request('POST','http://cbseaff.nic.in/cbse_aff/schdir_Report/userview.aspx',data = payload1,headers=hed1)
    school_page_soup = BeautifulSoup(school_page.content)
    
    table_t1 = school_page_soup.find('table',id='T1')
    table_with_cellpadding_1 = table_t1.find_all('table',cellpadding=1)
    #td_center= table_t1.find_all('td', {'align': 'left'})
    
    for e,i in enumerate(table_with_cellpadding_1):
        l=[]
        for j in i.find_all('td', {'align': 'left'}):
            l.append(re.sub('[\n\t\r]','',j.text))
        q.append(l)
    print "Page number " + str(z) + " is done out of 747"    


import pickle
with open('cbseaff', 'wb') as fp:
    pickle.dump(q, fp)
#    print ', '.join(l)

#transose dofferent list inside list and create another list
transpose=[]
for h in q:
    transpose.append(', '.join(h))
    
import csv

with open("cbseaff_allschools.csv",'wb') as resultFile:
    wr = csv.writer(resultFile, dialect='excel')
    wr.writerow([unicode(s).encode("utf-8") for s in transpose])  
  
  