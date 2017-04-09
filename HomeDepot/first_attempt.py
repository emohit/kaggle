# -*- coding: utf-8 -*-
"""
Created on Fri Jan 08 17:46:11 2016

@author: monarang
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import pandas as pd
from sklearn.grid_search import GridSearchCV
#from sklearn.feature_selection import SelectKModel,f_classif
import re
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer('english')

def str_stemmer(s):
	return " ".join([stemmer.stem(word) for word in s.lower().split()])

def str_common_word(str1, str2):
	return sum(int(str2.find(word)>=0) for word in str1.split())

root_path = "C:\\Deloitte\\Kaggle\\\HomeDepot\\"
train_data = pd.read_csv(root_path+"train.csv",encoding="ISO-8859-1")
#test_data = pd.read_csv(root_path+"test.csv",encoding="ISO-8859-1")
prod_desc = pd.read_csv(root_path+"product_descriptions.csv",encoding="ISO-8859-1")
#prod_attr = pd.read_csv(root_path+"attributes.csv",encoding="ISO-8859-1")

prod_desc.product_description=prod_desc.product_description.apply(lambda x : re.sub(r'([a-z]+)([A-Z]\.+)',r'\1 \2',x))
prod_desc.product_description=prod_desc.product_description.apply(lambda x : re.sub(r'([a-z]+)([A-Z0-9][a-z]+)',r'\1 \2',x))
prod_desc.product_description=prod_desc.product_description.apply(lambda x : re.sub(r'(\.)([a-zA-Z0-9]+)',r'\1 \2',x))
prod_desc.product_description=prod_desc.product_description.apply(lambda x : re.sub(r'([0-9]).?in\.',r'\1inches ',x))

train_data['search_in_desc']=train_data.apply(lambda x : set(x['search_term'].lower().split()).issubset(set(x['product_title'].lower().split())),axis=1)
train_data['search_inter_desc']=train_data.apply(lambda x : float(len(set(x.search_term.lower().split()).intersection (set(x.product_title.lower().split()))))/float(len(set(x.search_term.lower().split()))),axis=1)



prod_desc['product_description'] = prod_desc['product_description'].map(lambda x:str_stemmer(x))


train_prod_desc = pd.merge(train_data, prod_desc, how='left', on=['product_uid'])
#test_prod_desc = pd.merge(test_data, prod_desc, how='left', on=['product_uid'])

#aggr_prod_attr=prod_attr.groupby('product_uid')['value'].apply(lambda x: ' '.join(unicode(v) for v in x)).reset_index()

#train_desc_attr = pd.merge(train_prod_desc, aggr_prod_attr, how='left', on=['product_uid'])
#test_desc_attr = pd.merge(test_prod_desc, aggr_prod_attr, how='left', on=['product_uid'])

#train_desc_attr=train_desc_attr.drop(['id','product_uid'],axis=1)
#test_desc_attr=test_desc_attr.drop(['id','product_uid'],axis=1)

traindata = list(train_prod_desc.apply(lambda x:'%s %s %s' % (x['search_term'],x['product_title'], x['product_description']),axis=1))
#testdata = list(train_prod_desc.apply(lambda x:'%s %s %s' % (x['search_term'],x['product_title'], x['product_description']),axis=1))

del(train_data)
#del(test_data)
#del(prod_attr)
del(prod_desc)
del(train_prod_desc)
#del(test_prod_desc)
#del(aggr_prod_attr)
#del(train_desc_attr)
#del(test_desc_attr)


# the infamous tfidf vectorizer (Do you remember this one?)
tfv = TfidfVectorizer(min_df=3,  max_features=50000, 
        strip_accents='unicode', analyzer='word',token_pattern=r'(?u)\b[A-z]{3,}\b',
        ngram_range=(1,1), use_idf=1,smooth_idf=1,sublinear_tf=1,
        stop_words = 'english')

# Fit TFIDF
tfv.fit(traindata)
X =  tfv.transform(traindata) 


# LSA / SVD
svd = TruncatedSVD(n_components = 100)
X = svd.fit_transform(X)
X_test = svd.transform(X_test)