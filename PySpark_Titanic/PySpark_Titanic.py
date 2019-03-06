
# coding: utf-8

# In[1]:


from pyspark.sql import SparkSession


# In[2]:


df = spark.read.csv('dbfs:/FileStore/tables/train.csv', inferSchema=True, header=True)


# In[3]:


df.printSchema()


# In[4]:


print(df.count(), len(df.columns))


# In[5]:


df.show()


# In[6]:


df = df.drop('PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked', 'Sex')


# In[7]:


df.show()


# In[8]:


dftrain, dftest = df.randomSplit([0.8,0.2])


# In[9]:


dftrain.describe().show()
#df.describe('Age').show


# In[10]:


df.describe('Age').show()


# In[11]:


dftrain.select('Age', 'Pclass').show()


# In[12]:


dftrain.groupby('Pclass').count().show()


# In[13]:


dftrain.groupBy('Pclass').count().show()


# In[14]:


from pyspark.ml.feature import Imputer
impage = Imputer(strategy='median', inputCols=['Age'], outputCols=['impAge']).fit(dftrain)
dftrain=impage.transform(dftrain)
dftrain=dftrain.drop('Age')
dftrain.show()


# In[15]:


from pyspark.ml.feature import StringIndexer
sipclass = StringIndexer(handleInvalid='keep', inputCol='Pclass', outputCol='idxPclass').fit(dftrain)
dftrain=sipclass.transform(dftrain)
dftrain=dftrain.drop('Pclass')


# In[16]:


dftrain.show()


# In[17]:


from pyspark.ml.feature import OneHotEncoderEstimator
ohe=OneHotEncoderEstimator(handleInvalid='keep', dropLast=True, inputCols=['idxPclass'], outputCols=['ohePclass']).fit(dftrain)
dftrain=ohe.transform(dftrain)
dftrain=dftrain.drop('idxPclass')
dftrain.sample(withReplacement=False, fraction=0.1).limit(20).show()


# In[18]:


from pyspark.ml.feature import VectorAssembler
va = VectorAssembler(inputCols=['SibSp', 'Parch', 'Fare', 'impAge', 'ohePclass'],
                    outputCol='features')
dftrain=va.transform(dftrain)
dftrain = dftrain.drop('SibSp', 'Parch', 'Fare', 'impAge', 'ohePclass')
dftrain.show()


# In[19]:


from pyspark.ml.classification import RandomForestClassifier
rfc = RandomForestClassifier(labelCol="Survived", featuresCol="features", numTrees=100).fit(dftrain)


# In[20]:


dftest.show()


# In[21]:


dftest = impage.transform(dftest)
dftest = dftest.drop('Age')


# In[22]:


dftest = sipclass.transform(dftest)
dftest = dftest.drop('Pclass')
dftest = ohe.transform(dftest)
dftest = dftest.drop('idxPclass')


# In[23]:


dftest = va.transform(dftest)
dftest = dftest.drop('SibSp','Parch','Fare','impAge','ohePclass')
dftest.show()


# In[24]:


# predict using random forest classifier on test data
predictions = rfc.transform(dftest)
predictions.show()


# In[25]:


# evaluate prediction results
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator(
labelCol="Survived",
predictionCol="prediction",
metricName="accuracy")
evaluator.evaluate(predictions)

