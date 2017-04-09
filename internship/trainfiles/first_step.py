# -*- coding: utf-8 -*-
"""
Created on Fri Jan 08 17:46:11 2016

@author: monarang
"""
from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor,AdaBoostClassifier
from sklearn.ensemble  import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.grid_search import GridSearchCV
from sklearn import svm
#from sklearn.feature_selection import SelectKModel,f_classif
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest,chi2,f_classif

dateparse = lambda x: pd.datetime.strptime(x, '%d-%m-%Y') if(str(x).find('-')) > 0 else pd.datetime.strptime(x, '%m/%d/%Y')

root_path = "C:\\Deloitte\\Kaggle\\internship\\trainfiles\\"
intern_data = pd.read_csv(root_path+"Internship\\Internship.csv",parse_dates=['Start_Date','Internship_deadline'],date_parser=dateparse)
student_data = pd.read_csv(root_path+"Student\\Student.csv")
train_data = pd.read_csv(root_path+"traincsv\\train.csv",parse_dates=['Earliest_Start_Date'],date_parser=dateparse)
test_data = pd.read_csv("C:\\Deloitte\\Kaggle\\internship\\test-date-your-data\\test.csv",parse_dates=['Earliest_Start_Date'],date_parser=dateparse)

#handling student profile
student_profile=student_data[['Student_ID','Profile']]

student_profile_grp=student_profile.groupby('Student_ID')['Profile'].apply(lambda x: ''.join(str(set(x)))).reset_index()

#handling non student profile
grp=student_data.groupby([u'Student_ID', u'Institute_Category', u'Institute_location', u'hometown', u'Degree', u'Stream', u'Current_year', u'Year_of_graduation', u'Performance_PG', u'PG_scale', u'Performance_UG', u'UG_Scale', u'Performance_12th', u'Performance_10th'],as_index='false')
q=grp.groups
student_non_profile=pd.DataFrame(q.keys(),columns=[u'Student_ID', u'Institute_Category', u'Institute_location', u'hometown', u'Degree', u'Stream', u'Current_year', u'Year_of_graduation', u'Performance_PG', u'PG_scale', u'Performance_UG', u'UG_Scale', u'Performance_12th', u'Performance_10th'])

student_profile_grp=pd.merge(student_non_profile,student_profile_grp,on='Student_ID')

train_student = pd.merge(train_data,student_profile_grp,on='Student_ID')
test_student = pd.merge(test_data,student_profile_grp,on='Student_ID')

#handling the intership ID data to concatenate the profiles where =1 and merging again with original data
sparse_matrix=intern_data[['Internship_ID','PR','UI','Marketing','Media','Social','Design','Web','Development','Business','Research','Writing','Plan','Creative','Process','Database','Strategy','Designing','Analysis','Facebook','Communication','Rest','Android','Presentation','Media Marketing','Twitter','Social Media Marketing','Operations','Java','Quality','HTML','Blogs','Digital Marketing','PHP','Market Research','Recruitment','Testing','CSS','Planning','API','Editing','Content Writing','Innovative','Lead Generation','Mobile App','SQL','Accounts','Reporting','JavaScript','Documentation','iOS','Branding','ACTING','Analytics','Initiative','Advertising','Cold Calling','Sourcing','ERP','NGO','Environment','Networking','Production','MySQL','ISO','Marketing Strategy','Survey','Visio','App Development','Front End','web development','Integration','HTML5','jQuery','Server','Coding','MBA','Content Creation','Reading','B2B','Content Development','Storm','E-commerce','Databases','Programming','Wordpress','Outreach','NABL','Web Design','Architecture','Web Application','Adobe','Scala','UI/UX','Python','Mac','Retail','Digital Media','Product Development','Data Collection','Algorithm','LESS','Email Marketing','Screening','Bootstrap','Finance','Content Marketing','Communication Skil','Hiring','Negotiation','Administration','Communication Skills','CSS3','Infographic','Youtube','CRM','CAD','Infographics','Access','Editorial','ARM','AJAX','.NET','Co-ordination','Ownership','Algorithms','Node','Drafting','Blogging','Animation','Teaching','Blogger','Relationship Management','3d','HTTP','Press Release','Accounting','Android App Development','Adobe Photoshop','Photography','Software Development','Social Networking','AngularJS','AWS','Secondary Research','Recruiting','Client Servicing','Leadership','Content Writer','Web Services','Payroll','Prospecting','Graphic Designing','Proofreading','Data Entry','Flex','Creativity','Data Management','Convincing','GATE','Social Media Management','Machine Learning','Client Relations','Web Applications','XML','MVC','HTML/CSS','Google+','Typing','Sketch','UI Design','Visual Design','Creative Writing','Graphic Designer','Product Design','PERL','Hindi','Chef','Sales Process','ASP.NET','Django','Public Relations','CMS','Vendor Management','Content Strategy','Client Relationship','Creative Design','C#','JSON','Linux','Client Interaction','Manufacturing','Customer Relationship Management','Recruitment Process','Business Relation','Talent Acquisition','CorelDRAW','Big Data','Material Design','Market Analysis','Adobe Illustrator','REST API','Tally','Electronics','Bee','C++','Online Research','Mockups','Front End Development','Gif','Product Management','MongoDB','Primary Research','Healthcare','Data Analytics','Google Analytics','Core PHP','B2B Sales','Social Media Tools','Node.js','Ruby','Drawing','Brand Promotion','Mechanical','Automobile','Lifestyle','Writing Blogs','CodeIgniter','Writing Skills','SQL Server','Logo Design','Project Management','API Integration','Client Communication','Growth Hacking','Interior Design','Personality','SAP','Scripting','Android Application Development','Event Management','Blog Writing','Statistics','Typography','Scalability','Assembly','Conceptualization','Microsoft','MVC Framework','PSD','Web Technologies','Web Application Development','Joomla','Image Processing','REST APIs','Data Structure','Confidence','Electrical','Counseling','Inside Sales','Organizational Skills','Video Editing','Data Structures','Mobile Application Development','PhoneGap','Storytelling','MySQL.','Ionic','Design Skills','Corporate Sales','Entrepreneurship','Films','Foundation','Payment Gateway']]
non_sparse_matrix=intern_data.iloc[:,:12]

sub={}
for i in range(len(sparse_matrix)):
    a=''
    for j in sparse_matrix.columns:
        if sparse_matrix[j][i]==1:
            a+=' '+j            
        sub[sparse_matrix.Internship_ID[i]]=a.lstrip()

new_df = pd.DataFrame(sub.items(),columns=['Internship_ID','skills'])

del(sub)
del(sparse_matrix)
#concatenating with the original data 
new_intern_df = pd.merge(new_df,non_sparse_matrix,on='Internship_ID')

del(intern_data)
#end of the functionality
#need to work on the student.csv as it contain mutiple entry of the student id with different streams
#student_profile=student_data[['Student_ID','Profile']]

#student_profile_grp=student_profile.groupby('Student_ID')['Profile'].apply(lambda x: ''.join(str(set(x)))).reset_index()

full_df = pd.merge(train_student,new_intern_df,on='Internship_ID')
full_df.Profile=full_df.Profile.apply(lambda x : x.replace('set([','').replace('])','').replace(',',''))

#del(student_data)
del(train_data)
full_df.Profile=full_df.Profile.apply(lambda x : str(x).replace('\'','').replace('nan','').lstrip())

#full_df.to_csv(root_path+"full_df.csv", index=False)

#full_df=pd.read_csv(root_path+"full_df.csv")
#full_df['is_eligible']=full_df.apply(lambda x :1 if x['Year_of_graduation']>x['Start_Date'][:4] else 0,axis=1)
full_df['is_eligible_start_date']=full_df.apply(lambda x : 1 if x['Earliest_Start_Date'] < x['Internship_deadline'] else 0 ,axis=1)
full_df['is_eligible_start_date1']=full_df.apply(lambda x : 1 if x['Earliest_Start_Date'] >= x['Start_Date'] else 0 ,axis=1)

#earliest start date
full_df['earliest_day']=full_df.Earliest_Start_Date.apply(lambda x :int(str(x)[-2:]) if len(str(x)) > 4 else 0)
full_df['earliest_month']=full_df.Earliest_Start_Date.apply(lambda x :int(str(x)[5:7]) if len(str(x)) > 4 else 0)
full_df['earliest_year']=full_df.Earliest_Start_Date.apply(lambda x :int(str(x)[:4]) if len(str(x)) > 4 else 0)

#Internship_deadline
full_df['Internship_deadline_day']=full_df.Internship_deadline.apply(lambda x :int(str(x)[-2:]) if len(str(x)) > 4 else 0)
full_df['Internship_deadline_month']=full_df.Internship_deadline.apply(lambda x :int(str(x)[5:7]) if len(str(x)) > 4 else 0)
full_df['Internship_deadline_year']=full_df.Internship_deadline.apply(lambda x :int(str(x)[:4]) if len(str(x)) > 4 else 0)

#Start_Date
full_df['Start_Date_day']=full_df.Start_Date.apply(lambda x :int(str(x)[-2:]) if len(str(x)) > 4 else 0)
full_df['Start_Date_month']=full_df.Start_Date.apply(lambda x :int(str(x)[5:7]) if len(str(x)) > 4 else 0)
full_df['Start_Date_year']=full_df.Start_Date.apply(lambda x :int(str(x)[:4]) if len(str(x)) > 4 else 0)


full_df['skill_in_profile']=full_df.apply(lambda x : len(set(str(x['Profile']).lower().replace('/',' ').split()).intersection(set(str(x['Internship_Profile']+' '+x['skills']).lower().replace('/',' ').split()))),axis=1)
full_df['skill_in_inter_profile']=full_df.apply(lambda x : len(set(str(x['skills']).lower().replace('/',' ').split()).intersection(set(str(x['Internship_Profile']).lower().replace('/',' ').split()))),axis=1)
#full_df['esd_year']=full_df.Earliest_Start_Date.apply(lambda x : str(x)[-4:])
#full_df['esd_year_len']=full_df.Earliest_Start_Date.apply(lambda x : 1 if len(str(x))==5 else 0)
full_df['bol_loc']=full_df.apply(lambda x :len( set(str(x['Preferred_location']).lower()).intersection(set(str(x['Internship_Location']).lower()))) if len(str(x['Preferred_location']))==4 and len(str(x['Internship_Location']))==4 else '0',axis=1)
full_df['bol_loc1']=full_df.apply(lambda x :len( set(str(x['Institute_location']).lower()).intersection(set(str(x['Internship_Location']).lower()))) if len(str(x['Institute_location']))==4 and len(str(x['Internship_Location']))==4 else '0',axis=1)
full_df['bol_loc2']=full_df.apply(lambda x :len( set(str(x['hometown']).lower()).intersection(set(str(x['Internship_Location']).lower()))) if len(str(x['hometown']))==4 and len(str(x['Internship_Location']))==4 else '0',axis=1)
full_df.Performance_PG = full_df.Performance_PG /full_df.PG_scale
full_df.Performance_UG = full_df.Performance_UG /full_df.UG_Scale

full_df.to_csv(root_path+"full_df.csv", index=False)

le = preprocessing.LabelEncoder()

full_df.Expected_Stipend = le.fit_transform(full_df.Expected_Stipend)
full_df.Internship_Type = le.fit_transform(full_df.Internship_Type)
full_df.Internship_category = le.fit_transform(full_df.Internship_category)
full_df.Stipend_Type = le.fit_transform(full_df.Stipend_Type)
full_df.Institute_Category = le.fit_transform(full_df.Institute_Category)
full_df.Current_year = le.fit_transform(full_df.Current_year)
full_df.Year_of_graduation = le.fit_transform(full_df.Year_of_graduation)


#is_eligible,Current_year,esd_year,esd_year_len,,u'skill_in_profile','earliest_day','earliest_month','earliest_year','Internship_deadline_day','Internship_deadline_month','Internship_deadline_year','Start_Date_day','Start_Date_month','Start_Date_year',Expected_Stipend,,'Performance_10th','Performance_12th','Minimum_Duration',,'Year_of_graduation','Performance_PG','Performance_UG'
column_name=['skill_in_profile','skill_in_inter_profile','is_eligible_start_date1','is_eligible_start_date','Institute_Category','bol_loc','bol_loc1','bol_loc2','Stipend_Type','Internship_category','Internship_Type','Is_Part_Time']
X_imputed_df = full_df[column_name]
y = full_df.Is_Shortlisted

#Modelling
gbm = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,max_depth=3, random_state=0)
clf = RandomForestClassifier(n_estimators=10)
svr = svm.SVC()
etc = AdaBoostClassifier(n_estimators=100,learning_rate=0.1, random_state=0)

parameters_gbm = {'n_estimators':[10],'learning_rate':[0.01],'max_depth':[1],'min_samples_split':[1]}
parameters_clf = {'n_estimators':[50]}
parameters = {'kernel':['rbf','poly'], 'C':[0.01,0.05,0.1,0.5], 'gamma': [0.1,0.01,1e-3]}
parameters_etc = {'n_estimators':[50,100,150],'learning_rate':[0.01,0.1,0.5,1.0]}

gscv = GridSearchCV(clf, parameters_clf,cv=3,verbose=5)
                       
gscv.fit(X_imputed_df, y)

#selector_chi = SelectKBest(chi2,4)
#selector_chi.fit(X_imputed_df,y)
#
#selector_f = SelectKBest(f_classif,4)
#selector_f.fit(X_imputed_df,y)
#
#print selector_chi.pvalues_
#print selector_f.pvalues_


print "column used : " + str(X_imputed_df.columns)
print "Best estimator : "+ str(gscv.best_estimator_)
print "Best score of the estimator : "+  str(gscv.best_score_)

print "sum of prediction: "+str(sum(gscv.predict(X_imputed_df)))


#test manupulating starts from here 
full_df_test = pd.merge(test_student,new_intern_df,on='Internship_ID')
full_df_test.Profile=full_df_test.Profile.apply(lambda x : x.replace('set([','').replace('])','').replace(',',''))

#del(student_data)
full_df_test.Profile=full_df_test.Profile.apply(lambda x : str(x).replace('\'','').replace('nan','').lstrip())

#full_df_test.to_csv(root_path+"full_df_test.csv", index=False)

#full_df_test=pd.read_csv(root_path+"full_df_test.csv")
#full_df_test['is_eligible']=full_df.apply(lambda x :1 if x['Year_of_graduation']>x['Start_Date'][-4:] else 0,axis=1)
full_df_test['is_eligible_start_date']=full_df_test.apply(lambda x : 1 if x['Earliest_Start_Date'] < x['Internship_deadline'] and x['Earliest_Start_Date'] >= x['Start_Date'] else 0 ,axis=1)
full_df_test['is_eligible_start_date1']=full_df_test.apply(lambda x : 1 if x['Earliest_Start_Date'] >= x['Start_Date'] else 0 ,axis=1)


#earliest start date
full_df_test['earliest_day']=full_df_test.Earliest_Start_Date.apply(lambda x :int(str(x)[-2:]) if len(str(x)) > 4 else 0)
full_df_test['earliest_month']=full_df_test.Earliest_Start_Date.apply(lambda x :int(str(x)[5:7]) if len(str(x)) > 4 else 0)
full_df_test['earliest_year']=full_df_test.Earliest_Start_Date.apply(lambda x :int(str(x)[:4]) if len(str(x)) > 4 else 0)

#Internship_deadline
full_df_test['Internship_deadline_day']=full_df_test.Internship_deadline.apply(lambda x :int(str(x)[-2:]) if len(str(x)) > 4 else 0)
full_df_test['Internship_deadline_month']=full_df_test.Internship_deadline.apply(lambda x :int(str(x)[5:7]) if len(str(x)) > 4 else 0)
full_df_test['Internship_deadline_year']=full_df_test.Internship_deadline.apply(lambda x :int(str(x)[:4]) if len(str(x)) > 4 else 0)

#Start_Date
full_df_test['Start_Date_day']=full_df_test.Start_Date.apply(lambda x :int(str(x)[-2:]) if len(str(x)) > 4 else 0)
full_df_test['Start_Date_month']=full_df_test.Start_Date.apply(lambda x :int(str(x)[5:7]) if len(str(x)) > 4 else 0)
full_df_test['Start_Date_year']=full_df_test.Start_Date.apply(lambda x :int(str(x)[:4]) if len(str(x)) > 4 else 0)


full_df_test['skill_in_profile']=full_df_test.apply(lambda x : len(set(str(x['Profile']).lower().replace('/',' ').split()).intersection(set(str(x['Internship_Profile']+' '+x['skills']).lower().replace('/',' ').split()))),axis=1)
full_df_test['skill_in_inter_profile']=full_df_test.apply(lambda x : len(set(str(x['skills']).lower().replace('/',' ').split()).intersection(set(str(x['Internship_Profile']).lower().replace('/',' ').split()))),axis=1)
#full_df_test['esd_year']=full_df_test.Earliest_Start_Date.apply(lambda x : x[-4:])
#full_df_test['esd_year_len']=full_df_test.Earliest_Start_Date.apply(lambda x : 1 if len(x)==5 else 0)
full_df_test['bol_loc']=full_df_test.apply(lambda x :len( set(str(x['Preferred_location']).lower()).intersection(set(str(x['Internship_Location']).lower()))) if len(str(x['Preferred_location']))==4 and len(str(x['Internship_Location']))==4 else '0',axis=1)
full_df_test['bol_loc1']=full_df_test.apply(lambda x :len( set(str(x['Institute_location']).lower()).intersection(set(str(x['Internship_Location']).lower()))) if len(str(x['Institute_location']))==4 and len(str(x['Internship_Location']))==4 else '0',axis=1)
full_df_test['bol_loc2']=full_df_test.apply(lambda x :len( set(str(x['hometown']).lower()).intersection(set(str(x['Internship_Location']).lower()))) if len(str(x['hometown']))==4 and len(str(x['Internship_Location']))==4 else '0',axis=1)

full_df_test.Expected_Stipend = le.fit_transform(full_df_test.Expected_Stipend)
full_df_test.Internship_Type = le.fit_transform(full_df_test.Internship_Type)
full_df_test.Internship_category = le.fit_transform(full_df_test.Internship_category)
full_df_test.Stipend_Type = le.fit_transform(full_df_test.Stipend_Type)
full_df_test.Institute_Category = le.fit_transform(full_df_test.Institute_Category)
full_df_test.Current_year = le.fit_transform(full_df_test.Current_year)

test_imputed_df=full_df_test[column_name]

predictions=gscv.predict(test_imputed_df)
print sum(predictions)

submission = pd.DataFrame({ 'Internship_ID': full_df_test['Internship_ID'],
                            'Student_ID' : full_df_test['Student_ID'],
                            'Is_Shortlisted' : predictions })
submission.to_csv(root_path+"submission_svm.csv", index=False)