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
first_page = s.get("http://www.ebay.in/itm/SH3-Sonoff-Wifi-Smart-Switch-for-Home-Automation-with-Android-iOS-Free-App-ITEAD-/272283651660?hash=item3f655ece4c:g:Ac0AAOSw3YNXa~QT")

soup = BeautifulSoup(first_page.content)

#def page_scrap_for_sold(soup):
try:
    product_desc = soup.find('h1', class_="it-ttl").text
    average_views = soup.find('div', class_="vi-notify-new-bg-dBtm").text
    available = soup.find('span', {'id':'qtySubTxt'}).text
    sold = soup.find('span', class_ = "vi-qtyS vi-bboxrev-dsplblk vi-qty-vert-algn vi-qty-pur-lnk").text
    sold_href = soup.find('span', class_ = "vi-qtyS vi-bboxrev-dsplblk vi-qty-vert-algn vi-qty-pur-lnk").find('a').get('href')               
    store_name = soup.find('div', class_="mbg").text
    store_url = soup.find('div', class_="mbg").a.get('href')                          
    #return product_desc,average_views,available,sold,sold_href
except:
    pass

for start in range(0,1):
    url = "http://www.google.co.in/search?q=site:ebay.in+%22More+than+10+available+/+*+sold%22&start=" + str(start*10)
    google_first_page = s.get(url)

    google = BeautifulSoup(google_first_page.content)
    for i in google.findAll("div",class_="g"):
        print i.text.split('|')[0],i.cite