# your code here
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# In[2]:
# ### 2) Read the heart.csv file into a dataframe called df, and  view a 2% random sample of the data
# In[3]:
# your code here
df=pd.read_csv("heart.csv")
df.sample(frac=0.02)
# In[4]:
# ### 3) Split the data into Xtrain, Xtest, ytrain, ytest - with 30% in test, and random_state=1
# In[5]:
# your code here
from sklearn.model_selection import train_test_split
Xtrain,Xtest,ytrain,ytest = train_test_split(df.drop(['target'], axis=1), df['target'], random_state=1, test_size=0.3)
# In[6]:
# ### 4) Create two lists called numeric_features (with age, trestbps, chol, thalach, oldspeak), and categorical_features (with sex, cp, fbs, restecg, exang, slope, ca, thal)
# In[7]:
# your code here
numeric_features=['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
categorical_features=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
# In[8]:
# ### 5) Get basic stats on Xtrain (such as count, mean, std, etc)
# In[9]:
# your code here
Xtrain.describe()
# In[10]:
# ### 6) Check if there are any NaNs in any of the columns in Xtrain
# In[11]:
# your code here
Xtrain.isna().sum()
# In[12]:
# ### 7) Create an annotated heatmap of the correlation between the numeric features in Xtrain
# In[13]:
# your code here
sns.heatmap(Xtrain[numeric_features].corr(), annot=True)
# In[14]:
# ### 8) Standard Scale all of the numeric features in Xtrain, include transformed numeric features in Xtrain, and drop original numeric features in Xtrain
# In[15]:
# your code here
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
Xtrain_ss = pd.DataFrame(ss.fit_transform(Xtrain[numeric_features]), index=Xtrain.index, columns=['ss_'+col for col in numeric_features])
Xtrain = pd.concat([Xtrain, Xtrain_ss], axis=1)
Xtrain = Xtrain.drop(numeric_features, axis=1)
# In[16]:
# ### 9) OneHotEncode all of the categorical features in Xtrain, include transformed categorical features in Xtrain, and drop original categorical features in Xtrain
# In[17]:
# your code here
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False, dtype=int, handle_unknown='ignore')
Xcat = pd.DataFrame(ohe.fit_transform(Xtrain[categorical_features]), columns=ohe.get_feature_names(), index=Xtrain.index)
Xtrain = pd.concat([Xtrain, Xcat], axis=1)
Xtrain = Xtrain.drop(categorical_features, axis=1)
# In[18]:
# ### 10) Fit a Logistic Regression Model to training data with random_state=1
# In[19]:
# your code here
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=1)
lr.fit(Xtrain,ytrain)
# In[20]:
# ### 11) Standard Scale all of the numeric features in Xtest, include transformed numeric features in Xtest, and drop original numeric features in Xtest
# In[21]:
# your code here
Xtest_ss = pd.DataFrame(ss.transform(Xtest[numeric_features]), index=Xtest.index, columns=['ss_'+col for col in numeric_features])
Xtest = pd.concat([Xtest, Xtest_ss], axis=1)
Xtest = Xtest.drop(numeric_features, axis=1)
# In[22]:
# ### 12) OneHotEncode all of the categorical features in Xtest, include transformed categorical features in Xtest, and drop original categorical features in Xtest
# In[23]:
# your code here
XcatTest = pd.DataFrame(ohe.transform(Xtest[categorical_features]), columns=ohe.get_feature_names(), index=Xtest.index)
Xtest = pd.concat([Xtest, XcatTest], axis=1)
Xtest = Xtest.drop(categorical_features, axis=1)
# In[24]:
# ### 13) Predict and Evaluate Logisitic Regression Model on Xtest
# In[25]:
# your code here
ypred = lr.predict(Xtest)
from sklearn import metrics
print (metrics.accuracy_score(ytest, ypred))
print (metrics.confusion_matrix(ytest, ypred))
print (metrics.classification_report(ytest, ypred))
# In[26]:
# <h1><center>Part B</center></h1>
# ### 14) (a) Read heart.csv file into a dataframe called df; (b) Split the data into Xtrain, Xtest, ytrain, ytest - with 30% in test, and random_state=1; (c) Create two lists called numeric_features (with age, trestbps, chol, thalach, oldspeak), and categorical_features (with sex, cp, fbs, restecg, exang, slope, ca, thal)
# In[27]:
# your code here
import numpy as np
import pandas as pd
df=pd.read_csv("heart.csv")
from sklearn.model_selection import train_test_split
Xtrain,Xtest,ytrain,ytest = train_test_split(df.drop(['target'], axis=1), df['target'], random_state=1, test_size=0.3)
numeric_features=['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
categorical_features=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
# In[28]:
# ### 15) Create a pipeline called "numeric_transformer" with a StandardScaler step called "ss" (use the same parameters that you used in Part A above)
# In[29]:
# your code here
from sklearn.pipeline import Pipeline
numeric_transformer = Pipeline(steps=[('ss', StandardScaler())])
# In[30]:
# ### 16) Create a pipeline called "categorical_transformer" with a OneHotEncoder step called "ohe" (use the same parameters that you used in Part A above)
# In[31]:
# your code here
categorical_transformer = Pipeline(steps=[('ohe', OneHotEncoder(sparse=False, dtype=int, handle_unknown='ignore'))])
# In[32]:
# ### 17) Create a column transformer called "preprocessor" with two transformers: (a) the first transformer called "num" which uses the numeric_transformer (that you defined above) on the numeric_features; and (b) the second transformer called "cat" which uses the categorical_transformer (that you defined above) on the categorical_features
# In[33]:
# your code here
from sklearn.compose import ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])
# In[34]:
# ### 18) Create a pipeline called "clf" with two steps: (a) the first step called "pp" which invokes the preprocessor you defined above; and (b) the second step called "lr" which involkes a logisitc regression model  (use the same parameters that you used in Part A above)
# In[35]:
# your code here
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
clf = Pipeline(steps=[('pp', preprocessor),
                      ('lr', LogisticRegression(random_state=1))])
# In[36]:
# ### 19) Fit the clf pipeline to the training data
# In[37]:
# your code here
clf.fit(Xtrain,ytrain)
# In[38]:
# ### 20) Predict and Evaluate clf pipeline on Xtest (you should end up with same results as in Part A above)
# In[39]:
# your code here
ypred=clf.predict(Xtest)
from sklearn import metrics
print (metrics.accuracy_score(ytest, ypred))
print (metrics.confusion_matrix(ytest, ypred))
print (metrics.classification_report(ytest, ypred))
# In[40]: