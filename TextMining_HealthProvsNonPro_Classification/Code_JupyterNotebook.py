
# coding: utf-8

# In[1]:


import os
import pandas as pd

os.listdir('HealthProNonPro')

def data2df (path, label):
    file, text = [], []
    for f in os.listdir(path):
        file.append(f)
        fhr = open (path+f, 'r', encoding='utf-8', errors='ignore')
        t = fhr.read()
        text.append(t)
        fhr.close()
    return (pd.DataFrame({'file': file, 'text': text, 'class':label}))

dfPro = data2df('HealthProNonPro\\Pro\\',1) # Neg
dfNonPro = data2df('HealthProNonPro\\NonPro\\',0) # Pos
df = pd.concat([dfPro, dfNonPro], axis=0)
df.sample(frac=0.0005)

# import train_test_split
from sklearn.model_selection import train_test_split
#Xtrain,Xtest,ytrain,ytest = train_test_split(df['text'], df['class'], test_size=0.3)
Xtrain,Xtest,ytrain,ytest = train_test_split(df['text'], df['class'], test_size=0.3)

#custom preprocessor
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
def preprocess(text):
    # replace one or more white-space characters with a space
    regex = re.compile(r"\s+")                               
    text = regex.sub(' ', text)    
    # lower case
    text = text.lower()          
    # remove digits and punctuation
    regex = re.compile(r"[%s%s]" % (string.punctuation, string.digits))
    text = regex.sub(' ', text)           
    # remove stop words
    sw = stopwords.words('english')
    text = text.split()                                              
    text = ' '.join([w for w in text if w not in sw]) 
    # remove short words
    ' '.join([w for w in text.split() if len(w) >= 2])
    # lemmatize
    text = ' '.join([(WordNetLemmatizer()).lemmatize(w) for w in text.split()]) 
    return text

#import pipeline, tfidf, naive_bayes, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV

#create Pipeline
clf=Pipeline(steps=[('pp', TfidfVectorizer(preprocessor=preprocess,
                                          use_idf=True, smooth_idf=True,
                                           min_df=1, max_df=1.0,
                                           max_features=None, ngram_range=(1, 1))),
                    ('mdl', MultinomialNB())])

#create parameters
parameters = {'pp__norm':('l1', 'l2', None), 'mdl__alpha':[0.01, 0.1, 0.2, 0.5, 1]}
#create GridSearch
grid_search = GridSearchCV(clf, parameters, cv=5, return_train_score=False)
#fit GridSearch
grid_search.fit(Xtrain, ytrain)

#predict using best model
ypred = grid_search.best_estimator_.predict(Xtest)
# print scores
from sklearn import metrics
print (metrics.accuracy_score(ytest, ypred))
print (metrics.confusion_matrix(ytest, ypred))
print (metrics.classification_report(ytest, ypred))

#print GridSerach best parameters
#print(grid_search.best_estimator_, "\n")
#print(grid_search.best_params_, "\n")
#print(grid_search.best_score_, "\n")
#print(grid_search.cv_results_, "\n")

