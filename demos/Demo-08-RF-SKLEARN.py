#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import os
import random as rand
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from models import custom_RF as crf


# ## Import data from data directory

# In[3]:


overall_lake_depth_data = pd.read_csv('load_dataset/LakeDepth/pts_merged_final.csv')
overall_lake_depth_data.head(5)


# Drop the FID and Data from the csv

# In[4]:


overall_lake_depth_nd = overall_lake_depth_data.drop(['FID', 'Date'], axis=1)
overall_lake_depth_nd.head(5)


# # Describe statistics to report anamolous data
# Look for anything weird

# In[5]:


overall_lake_depth_nd.describe()


# Let's convert the data in nparrays just for sklearn's sake
# (We won't do this in the RAPIDS env due to us having cuDF)

# In[7]:


labels = overall_lake_depth_nd['Depth_m']
covariate_spectral_bands = overall_lake_depth_nd.drop(['Depth_m'], axis=1)
spectral_bands_list = list(covariate_spectral_bands.columns)
#covariate_spectral_bands = np.array(covariate_spectral_bands)


# # Train - test splitting

# In[8]:


covariates_train, covariates_test, labels_train, labels_test = train_test_split(covariate_spectral_bands,
                                                                               labels, test_size = 0.2,
                                                                               random_state = 42)


# print('Training features shapes:', covariates_train.shape)
# print('Testing features shapes:', covariates_test.shape)
# print('Training labels shapes:', labels_train.shape)
# print('Testing labels shapes:', labels_test.shape)

# # Training

# In[9]:


rf_model = RandomForestRegressor(n_estimators=1000, random_state=42)


# In[10]:



rf_model.fit(covariates_train, labels_train)


# # Predictions

# In[11]:



predictions = rf_model.predict(covariates_test)
errors = abs(predictions - labels_test)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'meters.')


# Calculate mean absolute percentage error (MAPE)

# In[12]:


mape = 100 * (errors / labels_test)
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy,2),'%.')


# In[ ]:


crf.save_raw_model(rf_model, 'sklearn_model_0.sav')

