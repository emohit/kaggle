# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 14:45:39 2017

@author: monarang
"""

from bs4 import BeautifulSoup
import requests
import pandas as pd


s = requests.Session()
first_page = s.get("http://www.ebay.in/itm/DZ09-Bluetooth-Smart-Watch-with-Sim-Card-Camera-Android-iOS-Memory-Slot-/222484235452?hash=item33cd1804bc:g:xisAAOSw~FJZL-RJ")

soup = BeautifulSoup(first_page.content)

prod_title = soup.find("h1",class_="it-ttl")
prod_title = prod_title.text.split(u'\xa0')[1]
no_of_views=0

try:
    no_of_views=soup.find("div",class_="vi-notify-new-bg-dBtm")
    no_of_views=no_of_views.text.strip("\n")
except:
    pass

item_condition = soup.find("div",id="vi-itm-cond").text

item_price = soup.find("span",class_="notranslate").get('content')

is_hot=0
item_sold_url=0
try:
    item_sold = soup.find("span",class_="vi-bboxrev-dsplblk").text
    item_sold_url = soup.find("span",class_="vi-bboxrev-dsplblk").find('a').get('href')
    if soup.find("span",class_="vi-bboxrev-dsplblk").find(class_="vi-qtyS-hot-red"):
        is_hot=1
except:
    pass

quantity=[]
sell_date=[]
avg=[-1]

if item_sold_url:
    item_sold_content = s.get(item_sold_url)
    soup1 = BeautifulSoup(item_sold_content.content)
    td=soup1.find_all('td',class_='contentValueFont')
    for i,j in enumerate(td):
        if i%3==1:
            quantity.append(int(j.text))
        if i%3==2:
            sell_date.append(j.text.split(' ')[0].strip())
    seller_df = pd.DataFrame({'quantity':quantity,'date':sell_date})
    seller_df['date'] =pd.to_datetime(seller_df['date'])
    avg=seller_df.groupby('date').sum().mean()

print prod_title,no_of_views,item_condition,item_price,item_sold_url,avg[0],is_hot



