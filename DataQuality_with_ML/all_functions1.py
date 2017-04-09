# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 16:41:53 2016

@author: monarang
"""
import re
import nltk
from collections import Counter
from itertools import groupby

def is_special(s):
    if re.match(r'^\w+$', s):
        return 1
    else:
        return 0        

def is_number(s):
    if re.match(r'^[0-9]+$', s):
        return 1
    else:
        return 0   

def is_alphanumeric(s):
    if re.match(r'^[a-zA-Z0-9]+$', s):
        return 1
    else:
        return 0   
        
def string_len(s):
    return len(s)

def is_alpha(s):
    if re.match(r'^[a-zA-Z]+$', s):
        return 1
    else:
        return 0 

def is_special_start(s): #starting with special character
    if re.match(r'^[^a-zA-Z0-9].*', s):
        return 1
    else:
        return 0 
        
def is_special_end(s): #Ending with special character
    if re.match(r'^.*[^a-zA-Z0-9]$', s):
        return 1
    else:
        return 0 

def count_characters(s): #Count characters in the string
    return Counter(s)
    
def check_consecutive(word):
    count=1
    length=""
    char = []
    char_consecutive_count = []
    main = []
    sub = []
    albhabet_position = {'a':1,'b':2,'c':3,'d':4,'e':5,'f':6,'g':7,'h':8,'i':9,'j':10,'k':11,'l':12,'m':13,'n':14,'o':15,'p':16,'q':17,'r':18,'s':19,'t':20,'u':21,'v':22,'w':23,'x':24,'y':25,'z':26}
    char_repeat = [0]*26
    for i in range(1,len(word)):
        if word[i-1]==word[i]:
           count+=1
        else:
            count = 1
        if count > 1:
            char_repeat[albhabet_position[word[i-1]]] = count
    return char_repeat
#def count_consecutive(s):    
#    count=1
#    length=""
#    for i in range(1,len(word)):
#        if word[i-1]==word[i]:
#            count+=1
#        else:
#            count = 1
def count_consecutive(s):    
    ll=[]
    l = check_consecutive(s)
    ll.append(l)
    return ll


df = pd.read_csv("name2.csv")
print (df)
def create_training_data(df):
    temp0 = df['NAME'].apply(is_special)
    temp1 = df['NAME'].apply(is_number)
    temp2 = df['NAME'].apply(is_alphanumeric)
    temp3 = df['NAME'].apply(string_len)
    temp4 = df['NAME'].apply(is_alpha)
    temp5 = df['NAME'].apply(is_special_start)
    temp6 = df['NAME'].apply(is_special_end)
    #temp7 = df['NAME'].apply(count_characters)
    temp8 = df['NAME'].apply(count_consecutive)
#    xLower = df["NAME"].map(lambda x: x if type(x)!=str else x.lower())
    
  #  print xLower
#    print (temp8)
    temp9 = pd.DataFrame(temp8)
    l = pd.DataFrame(temp8)
#    print (l)
    columns = [df['NAME'], temp0, temp1, temp2, temp3, temp4, temp5, temp6, temp8]
    data1 = pd.concat(columns, axis=1)
#    print (data1)
    data1.to_csv('temp4.csv')

create_training_data(df)