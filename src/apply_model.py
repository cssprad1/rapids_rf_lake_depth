#!/usr/bin/env python
# coding: utf-8

# # Lake Depth RF to Raster Data

# System modules
import os
import time
from pprint import pprint

# Anything numeric related
import numpy as np
import matplotlib.pyplot as plt

# Anything geospatial related
import rasterio as rio

# Anything GPU/ML related
import cupy as cp
import cuml
import cudf
from models import custom_RF as crf


# ### Helper functions
def list_files(directory, extension):
    """ Helper function to read in files to list"""
    return list((f for f in os.listdir(directory) if f.endswith('.' + extension)))

def timer(start, end):
    """ Helper function for timing things """
    print (end - start)
    return (end - start)


# ### Useful file paths
TIF_PATH = '/att/nobackup/maronne/lake/forCaleb/'
TIF_FILES = list(list_files(TIF_PATH, 'tif'))
CURRENT_TIF = TIF_FILES[7]
pprint(TIF_FILES)

for i in range(7, len(TIF_FILES)):
    st_00 = time.time()
    CURRENT_TIF = TIF_FILES[i]
    print(CURRENT_TIF)
    print("One tif #", i)
    with rio.open(TIF_PATH+CURRENT_TIF) as raster_img:
        """
        The with statement is nice becuase it auto
        closes all rasterio tifs opened once below
        code is executed
        """
        n_cols = raster_img.width
        n_rows = raster_img.height
        n_bands = raster_img.count
        gt = raster_img.transform
        crs = raster_img.crs
        ndval = raster_img.nodata
        img_properties = (n_cols, n_rows, n_bands, gt, crs)
        
        st_0 = time.time()
        # Create numpy array to mimic tif
        img_nd = np.zeros((n_rows, n_cols, n_bands), np.float32)
        for b in range(n_bands):
            print(" - reading in band #", b)
            st_1 = time.time()
            img_nd[:, :, b] = raster_img.read(b+1) # Populate it with band pixel vals
            et_1 = time.time()
            timer(st_1, et_1)
        et_0 = time.time()
        timer(st_0, et_0)
    
    from pprint import pprint
    new_shape = (img_nd.shape[0] * img_nd.shape[1], img_nd.shape[2])
    img_nd_array = img_nd[:, :, :img_nd.shape[2]].reshape(new_shape)
    print(" - loading raster data to GPU")
    gpu_img = cp.asarray(img_nd_array)
    cdf_raster = cudf.DataFrame(gpu_img)
    n_rows_raster, n_cols_raster = cdf_raster.shape
    print(" - loading model")
    model_load = crf.load_model('best_test_03.sav')
    
    index_0 = n_rows_raster//2
    index_1 = index_0 * 2
    index_3 = n_rows_raster % 2 # Any indeces left out?
    print(" - starting predictions")
    predictions_1 = model_load.model.predict(cdf_raster[:index_0]) # Predict first half
    predictions_2 = model_load.model.predict(cdf_raster[index_0:]) # Predict other hals
    print(" - ending predictions")
    concat_predictions = cudf.concat([predictions_1, predictions_2], axis=0)
    
    array_predictions = concat_predictions.to_array()

    array_predictions = array_predictions.reshape(img_nd[:, :, 0].shape).astype(np.float32)

    array_predictions[img_nd[:, :, 0] == -9999.0] = ndval

    file_name_no_extension = CURRENT_TIF.split('.', 1)[0]
    print(file_name_no_extension)
    file_name_predicted = file_name_no_extension + '_predicted_0.tif'
    print(file_name_predicted)
    print(" - writing predictions to TIF :", file_name_predicted)
    with rio.open(file_name_predicted,
                   'w', 
                   driver='GTiff', 
                   height = array_predictions.shape[0],
                   width = array_predictions.shape[1],
                   count = 1,
                   dtype = array_predictions.dtype,
                   crs = crs,
                   transform = gt) as prediction_raster:
        prediction_raster.write(array_predictions, 1)
    et_00 = time.time()
    timer(st_00, et_00)
