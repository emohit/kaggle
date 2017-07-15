# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 16:53:44 2017

@author: monarang
"""

from bs4 import BeautifulSoup
import requests
import pandas as pd


s = requests.Session()
first_page = s.get("http://www.ebay.in/sch/allcategories/all-categories")

soup = BeautifulSoup(first_page.content)

prod_cat_link = soup.findAll("a",class_="clh")
cat_links = [i.get('href') for i in prod_cat_link]
prod_cat_name = [i.text for i in prod_cat_link]


#sub category
def get_sub_category(link,name):
    first_page = s.get("http://www.ebay.in/sch/AA-AAA-Batteries/181523/i.html")
    soup = BeautifulSoup(first_page.content)
    sub_sub_category_tag = soup.findAll("div",class_="cat-link")
    sub_sub_category_link = [i.find('a').get('href') for i in sub_sub_category_tag if i.find('a').text <> name]
    sub_sub_category_name = [i.find('a').text for i in sub_sub_category_tag if i.find('a').text <> name]
    return  sub_sub_category_link,sub_sub_category_name
 
all_links=[]    
for j,k in zip(cat_links,prod_cat_name):
    for p,q in zip(get_sub_category(j,k)[0],get_sub_category(j,k)[1]):
        all_links.append(zip(get_sub_category(p,q)[0],get_sub_category(p,q)[1]))

for i in all_links:
    for j in  i:
        