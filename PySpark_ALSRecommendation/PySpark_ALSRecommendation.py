
# coding: utf-8

# In[1]:


from pyspark.sql import SparkSession


# In[2]:


get_ipython().run_line_magic('fs', 'ls dbfs:/FileStore/tables')


# In[3]:


data = spark.read.csv('dbfs:/FileStore/tables/ratings.csv', inferSchema=True, header=True)


# In[4]:


data.show()
data.describe()


# In[5]:


# train test split
train_data, test_data = data.randomSplit([0.8,0.2])


# In[6]:


# instantiate model
from pyspark.ml.recommendation import ALS
als = ALS(maxIter=5,regParam=0.01,userCol='userId',itemCol='movieId',ratingCol='rating',coldStartStrategy='drop')


# In[7]:


# train
als_model = als.fit(train_data)


# In[8]:


# test
predictions = als_model.transform(test_data)
predictions.show()


# In[9]:


# evaluate
from pyspark.ml.evaluation import RegressionEvaluator
evaluator = RegressionEvaluator(metricName='rmse',labelCol='rating',predictionCol='prediction')
rmse = evaluator.evaluate(predictions)
rmse


# In[10]:


# make predictions for single user
single_user = test_data.filter(test_data['userId']==1).select(['movieId','userId'])
single_user.show()
recommendations = als_model.transform(single_user)
recommendations.orderBy('prediction',ascending=False).show()

