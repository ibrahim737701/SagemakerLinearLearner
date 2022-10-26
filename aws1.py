#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# In[2]:


df = pd.read_csv('housing_data.csv')
df.head(7)


# In[3]:


df.isnull().sum()


# In[4]:


X = df.drop(labels=['price'], axis=1)
y = df['price']


# In[6]:


X = X.astype('float32')
y = y.astype('float32')


# In[7]:


print(X.head())


# In[8]:


type(X)


# In[9]:


X = X.to_numpy()
y = y.to_numpy()


# In[12]:





# In[11]:


type(y)


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=108)


# In[14]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[15]:


import boto3
import sagemaker
from sagemaker.session import Session
from sagemaker.amazon.amazon_estimator import get_image_uri


# In[16]:


my_region = boto3.session.Session().region_name


# In[17]:


s3 =boto3.resource('s3')


# In[18]:


bucket_name = 'bucket-housing-data'

s3 = boto3.resource('s3')
try:
    if my_region == boto3.session.Session().region_name:
        s3.create_bucket(Bucket = bucket_name,
                         CreateBucketConfiguration = {
                             'LocationConstraint':'ap-south-1'
                         })
        print(f' Bucket {bucket_name} created')
except Exception as e:
    print(e)


# In[19]:


prefix = 'linear-model'
output_location = f's3://{bucket_name}/{prefix}/output'


# In[20]:


import io
import sagemaker.amazon.common as smac


# In[21]:


buf = io.BytesIO()
smac.write_numpy_to_dense_tensor(buf, X_train, y_train)
buf.seek(0)


# In[22]:


import os
bucket_name='bucket-housing-data'
prefix = 'linear-model'
key_train = 'train-data'


# In[23]:


boto3.resource('s3').Bucket(bucket_name).Object(os.path.join(prefix,'train', key_train)).upload_fileobj(buf)


# In[24]:


s3_train_data = f's3://{bucket_name}/{prefix}/train/{key_train}'
s3_train_data


# In[25]:


buf = io.BytesIO()
smac.write_numpy_to_dense_tensor(buf, X_test, y_test)
buf.seek(0)

bucket_name='bucket-housing-data'
prefix = 'linear-model'
key_test = 'test_data'
boto3.resource('s3').Bucket(bucket_name).Object(os.path.join(prefix,'test', key_test)).upload_fileobj(buf)

s3_test_data = f's3://{bucket_name}/{prefix}/test/{key_test}'


# In[26]:


from sagemaker.amazon.amazon_estimator import get_image_uri
container = get_image_uri(boto3.Session().region_name, 'linear-learner')
role = sagemaker.get_execution_role()
sagemaker_session = sagemaker.Session()


# In[27]:


linear_estimators = sagemaker.estimator.Estimator(container,
                                                 role,
                                                 instance_count = 1,
                                                 instance_type = 'ml.m4.xlarge',
                                                 output_path = output_location,
                                                 sagemaker_session = sagemaker_session)


# In[28]:


linear_estimators.set_hyperparameters(feature_dim = 6,
                                     predictor_type = 'regressor',
                                     mini_batch_size=100)


# In[29]:


linear_estimators.fit({'train': s3_train_data,
                        'validation': s3_test_data})


# In[30]:


linear_regression = linear_estimators.deploy(initial_instance_count=1,
                                            instance_type='ml.m4.xlarge')


# In[33]:


from sagemaker.predictor import CSVSerializer, JSONDeserializer
linear_regression.serializer = CSVSerializer()
linear_regression.deserializer = JSONDeserializer()


# In[34]:


test_results = linear_regression.predict(X_test)


# In[35]:


predictions = np.array([i['score'] for i in test_results['predictions']])


# In[39]:


df1 = pd.DataFrame({'Actual': y_test.flatten(),
                    'Predicted': predictions.flatten()   
})


# In[40]:


df1.head()


# In[ ]:




