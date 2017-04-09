import re
import nltk
from collections import Counter
from itertools import groupby
from nltk.corpus import wordnet
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
import matplotlib.pylab as plt
from sklearn.grid_search import GridSearchCV
import sys
from string import ascii_lowercase

def is_special(s):
        
        if pd.isnull(s) or re.match(r'^\w+$', s):
            return 1
        else:
            return 0   

        
def is_number(s):
    if pd.isnull(s) or re.match(r'^[0-9]+$', s):
        return 1
    else:
        return 0   

def is_alphanumeric(s):
    if pd.isnull(s) or re.match(r'^[a-zA-Z0-9]+$', s):
        return 1
    else:
        return 0   
        
def string_len(s):
    if pd.isnull(s):
        return 0
    else:
        return len(s)

def is_alpha(s):
    if pd.isnull(s) or re.match(r'^[a-zA-Z]+$', s):
        return 1
    else:
        return 0 

def is_special_start(s): #starting with special character
    if pd.isnull(s) or re.match(r'^[^a-zA-Z0-9].*', s):
        return 1
    else:
        return 0 
        
def is_special_end(s): #Ending with special character
    if pd.isnull(s) or re.match(r'^.*[^a-zA-Z0-9]$', s):
        return 1
    else:
        return 0 

def repeated(s):
    REPEATER = re.compile(r"(.+?)\1+$")
    match = REPEATER.match(s)
    return match.group(1) if match else None

def check_for_repeated_sequence(s): # it will check for repeated sequence such as "asasasasas" , it will output "as"
    sub = repeated(s)
    if sub:
        return 1
    else:
        return 0
    
def count_characters(s): #Count characters in the string
    return Counter(s)

def length_of_distinct(s): # length of distinct character in string 
    if pd.isnull(s):
        return 0
    else:
        return len(''.join(set(s)))

def checkSequence(s): #check whether a string has sequence of character like abc , abcd ,efghi
    if pd.isnull(s):
        return 0
    else:
        return 1 if (s in ascii_lowercase) else 0
    
def check_querty_Sequence(s): #check whether a string has sequence of character like abc , abcd ,efghi
    if pd.isnull(s):
        return 0
    else:
        return 1 if (s in 'qwertyuiopasdfghjklzxcvbnm') else 0  

def check_querty_keyboard_row(s): #check whether a string has sequence of character like abc , abcd ,efghi
    if pd.isnull(s):
        return 0
    elif(set(s).issubset('qwertyuiop')):
        return 1
    elif(set(s).issubset('asdfghjkl')):
        return 2
    elif(set(s).issubset('zxcvbnm')):
        return 3
    else:
        return 4

    
      
#not used
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
            char_repeat[albhabet_position[word[i]]] = count
    t  = ""
    for i in range(0,26):
        t = t + str(char_repeat[i]) + ","
   
    return t

#not used
def count_consecutive(s):    
    for i , j in groupby(s):
        p=len(list(j))
        if p > 1:
            return i,p

#Not yet used  
def get_human_names(text):
    tokens = nltk.tokenize.word_tokenize(text)
    pos = nltk.pos_tag(tokens)
    sentt = nltk.ne_chunk(pos, binary = False)
    person_list = []
    person = []
    name = ""
    for subtree in sentt.subtrees(filter=lambda t: t.label() == 'PERSON'):
        for leaf in subtree.leaves():
            person.append(leaf[0])
        if len(person) > 1: #avoid grabbing lone surnames
            for part in person:
                name += part + ' '
            if name[:-1] not in person_list:
                person_list.append(name[:-1])
            name = ''
        person = []

    return (person_list)
    
df = pd.read_csv("C:\Users\monarang\Documents\Projects\Kaggle\DataQuality_with_ML\Bad_name.csv",na_values={'Names': ' '})

df['is_special'] = df['Names'].apply(lambda x: is_special(x))
df['is_number'] = df['Names'].apply(lambda x: is_number(x))
df['is_alphanumberic'] = df['Names'].apply(lambda x: is_alphanumeric(x))
df['length'] = df['Names'].apply(lambda x: string_len(x))
df['is_alpha'] = df['Names'].apply(lambda x: is_alpha(x))
df['is_special_start'] = df['Names'].apply(lambda x: is_special_start(x))
df['is_special_end'] = df['Names'].apply(lambda x: is_special_end(x))
df['length_of_distinct'] = df['Names'].apply(lambda x: length_of_distinct(x))
df['is_sequence'] = df['Names'].apply(lambda x: checkSequence(x))
df['is_querty_Sequence'] = df['Names'].apply(lambda x: check_querty_Sequence(x))
df['is_querty_keyboard_row'] = df['Names'].apply(lambda x: check_querty_keyboard_row(x))

y = df['is_valid']
df = df.drop(['is_valid','Names'], axis=1)


gbm = GradientBoostingClassifier(n_estimators=300, learning_rate=0.1,max_depth=3, random_state=0)

parameters_gbm = {'n_estimators':[300],'learning_rate':[0.1],'max_depth':[5,7]}
gscv = GridSearchCV(gbm, parameters_gbm,cv=3,verbose=5)
                       
gscv.fit(df, y)

print "column used : " + str(df.columns)
print "Best estimator : "+ str(gscv.best_estimator_)
print "Best score of the estimator : "+  str(gscv.best_score_)
#print "Need less than 1319 RMSE :" + math.sqrt(-(gscv.best_score_))

df_test = pd.read_csv("C:\Users\monarang\Documents\Projects\Kaggle\DataQuality_with_ML\\test.csv",na_values={'Names': ' '})

df_test['is_special'] = df_test['Names'].apply(lambda x: is_special(x))
df_test['is_number'] = df_test['Names'].apply(lambda x: is_number(x))
df_test['is_alphanumberic'] = df_test['Names'].apply(lambda x: is_alphanumeric(x))
df_test['length'] = df_test['Names'].apply(lambda x: string_len(x))
df_test['is_alpha'] = df_test['Names'].apply(lambda x: is_alpha(x))
df_test['is_special_start'] = df_test['Names'].apply(lambda x: is_special_start(x))
df_test['is_special_end'] = df_test['Names'].apply(lambda x: is_special_end(x))
df_test['length_of_distinct'] = df_test['Names'].apply(lambda x: length_of_distinct(x))
df_test['is_sequence'] = df_test['Names'].apply(lambda x: checkSequence(x))
df_test['is_querty_Sequence'] = df_test['Names'].apply(lambda x: check_querty_Sequence(x))
df_test['is_querty_keyboard_row'] = df_test['Names'].apply(lambda x: check_querty_keyboard_row(x))


y_test = df_test['is_valid']
df_test_No_Names = df_test.drop(['is_valid','Names'], axis=1)


predictions=gscv.predict(df_test_No_Names)

print predictions
score = accuracy_score(predictions,y_test)
print " Accuracy of predicted and actual result is : " + str(float(score))


submission = pd.DataFrame({ 'Names': df_test['Names'],'is_valid': df_test['is_valid'],
                            'Predicted': predictions })
submission.to_csv("C:\Users\monarang\Documents\Projects\Kaggle\DataQuality_with_ML\submission.csv", index=False)