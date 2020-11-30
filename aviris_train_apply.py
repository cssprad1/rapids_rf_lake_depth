#!/usr/bin/env python
# coding: utf-8

# # Aviris 2020 random forest model training and testing
#
# ### What this notebook is doing.
#
# #### This notebook takes the AVIRIS data, trains a GPU-based model, and apply's the model to the raster stack
#
#
# 1. Data preprocessing- Load the Aviris data into the GPU, narrow down to the bands we want to sample
# 2. Model initialization / training - Initialize a GPU random forest model based on hyperparameters, train through k_fold, save model
# 3. Model raster application - Get the location of the tifs to apply the GPU RF model to, put them in a form (GPU ndarray) that's friendly to the model, then make batch predictions
# 4. Output the batch predictions back to an image shape, write using rasterio to a GTiff at the specified location.
#
#
# #### Things to know
#
# 1. Before running this make sure you make a directory 'saved_models' inside the 'models' directory
# 2. The models saved during training will overwrite themselves if they are trainied on the same nth band and starting index
# 3.  The rasters written at the end will overwrite themselves if the output path isn't changed, or the tifs aren't changed.
# 4.  If you want to keep every run's output rasters, do one of the above methods.

# In[1]:


import os
import logging
import time
from pprint import pprint

import numpy as np

from models.custom_RF import cuRF  # GPU/RAPIDS custom Random Forest objects
from load_dataset.aviris_dataset import Aviris  # Aviris is now a dataset object
# Facilitates training using sklearn's kfold
from src.k_fold_train import k_fold_train
# Handles all raster application, io, etc.
from src import apply_model_to_raster as amr


# In[2]:


logger = logging.getLogger(__name__)
logName = __name__+str(time.time())
logger.setLevel(logging.INFO)
logger.propagate = False
file_handler = logging.FileHandler(os.path.join('logs', logName))
formatter = logging.Formatter(
    '%(asctime)s : %(levelname)s : %(name)s : %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.info('Start logging for run')
print("Logging found in logs/"+logName)


# ## Control Panel
#
# This is where we can set what we need to set.
#
# Ideally we'll only need to change stuff in this cell and just run the others without changing them.
#
# - Make sure we make the changes before the cell is run

# In[3]:


NTH_BAND = 20  # Sample every N-th band from dataset.

START_INDEX = 0  # Index to start nth band sampling.

DATA_SPLIT_RATIO = 0.10  # Controls how much data is reserved for testing.

# Path to raster stacks to be predicted.
PATH = '/att/nobackup/maronne/AVIRIS_2020/stacks/'

# Controls the folds to use in k_fold cross validation training.
K_FOLD_FOLDS = 2

# Path for the prediction rasters
OUTPUT_PATH = '/att/nobackup/cssprad1/test_predictions/'

####################### logging ############################
logger.info('\nNTH_BAND:' + str(NTH_BAND) + '\n' + 'START_INDEX: ' + str(START_INDEX) + '\n' + 'Input path: ' +
            PATH + '\n'            'Output path: ' + OUTPUT_PATH + '\n' + 'k_folds: ' + str(K_FOLD_FOLDS) + '\n')
############################################################


# ## Data preprocessing
#
# We load the dataset object in, then sample every nth band using variables provided above.
#
# We then resplit the 'new' data.

# In[4]:


aviris_dataset = Aviris()
aviris_dataset.nth_row_sampling(START_INDEX, NTH_BAND)
print(aviris_dataset.covariates.columns)
aviris_dataset.split(DATA_SPLIT_RATIO)

# You can verify below that it samples the correct bands

####################### logging ############################
logger.info('\nBands sampled:' + str(aviris_dataset.covariates.columns) + '\n')
############################################################


# ## Hyperparameters for the random forest model
#
# Change as you see fit

# In[5]:


hyperparameters = {'N_ESTIMATORS': 1052,
                   'SPLIT_ALGO': 1,
                   'SPLIT_CRITERION': 2,
                   'BOOTSTRAP': True,
                   'BOOTSTRAP_FEATURES': False,
                   'ROWS_SAMPLE': 1.0,
                   'MAX_DEPTH': 30,
                   'MAX_LEAVES': -1,
                   'MAX_FEATURES': 'auto',
                   'N_BINS': 6,
                   'MIN_ROWS_PER_NODE': 2,
                   'MIN_IMPURITY_DECREASE': 0.0,
                   'ACCURACY_METRIC': 'mean_ae',  # 'mse' #'r2' # 'median_aw' #
                   'QUANTILEPT': False,
                   'SEED':  42,
                   'VERBOSE': False
                   }

####################### logging ############################
logger.info('\nModel Hyperparameters:' + str(hyperparameters) + '\n')
############################################################


# ## Random forest model
# - Initialization
# - Training
# - Metrics gathering

# In[6]:


rf_0 = cuRF(hyperparameters)


# ### We're going to use k_fold validation training to ensure some sort of protection against overfitting

# In[7]:


rf_0 = k_fold_train(K_FOLD_FOLDS,
                    rf_0,
                    NTH_BAND,
                    START_INDEX,
                    aviris_dataset.covariates_train,
                    aviris_dataset.labels_train)


# ### Metrics
#
# MAE: mean absolute error
# r2: r^2 score
# MSE: mean square error

# In[8]:


_, mae, r2, mse = rf_0.get_metrics(aviris_dataset.covariates_test,
                                   aviris_dataset.labels_test)

####################### logging ############################
logger.info('\nMAE: ' + str(mae) + '\n' + 'r2: ' +
            str(r2) + '\n' + 'MSE: ' + str(mse) + '\n')
############################################################

rf_0.feature_importances(aviris_dataset.covariates_train,
                         aviris_dataset.labels_train,
                         show=False,
                         nth_band=NTH_BAND,
                         starting_index=START_INDEX)
# # Raster processing
# - Raster loading
# - Reshaping to model-friendly format
# - Batch prediction
# - Writing predictions out as tif

# In[9]:


TIF_FILES = list(amr.list_files(PATH, 'tif'))
pprint(TIF_FILES)
# Make sure below that these are the tifs you want to apply the model to.
# If not, check your 'OUTPUT_PATH'.
####################### logging ############################
logger.info('\nTif files input:' + str(TIF_FILES))
############################################################
"""

# In[10]:


for index, TIF_FILE in enumerate(TIF_FILES):

    print("TIF prediction: ", index+1, "/", len(TIF_FILES))

    img_nd_array, img_nd_array_properties = amr.get_array_from_raster(PATH,
                                                                      TIF_FILE)

    img_nd_array_reshape = amr.change_img_shape(img_nd_array)
    img_nd_array_resample = img_nd_array_reshape[:, 36+START_INDEX::NTH_BAND]
    prediction_raster = amr.map_apply_reduce_pandas(img_nd_array_resample,
                                                    rf_0,
                                                    img_nd_array[:, :, 0].shape)

    amr.output_gtiff(prediction_raster,
                     img_nd_array_properties,
                     img_nd_array,
                     TIF_FILE,
                     OUTPUT_PATH)

    del img_nd_array, img_nd_array_reshape, img_nd_array_resample, prediction_raster

"""
# In[ ]:
