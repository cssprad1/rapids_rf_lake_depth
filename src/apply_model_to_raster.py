import os
import time

import numpy as np

import rasterio as rio
import cupy as cp
import cudf
import cuml
from models import custom_RF as crf
import pandas as pd


def list_files(directory, extension):
    """ Helper function to read in files to list"""

    return list((f for f in os.listdir(directory) if f.endswith('.' + extension)))


def timer(start, end):
    """ Helper function for timing things """

    print(end - start)
    return (end - start)


def get_array_from_raster(path, file):

    tif_path = path
    tif_file = file
    tif_filepath = os.path.join(tif_path, tif_file)

    print(tif_filepath)

    with rio.open(tif_filepath) as raster_img:
        img_ndarray = raster_img.read()
        n_cols = raster_img.width
        n_rows = raster_img.height
        n_bands = raster_img.count
        gt = raster_img.transform
        crs = raster_img.crs
        ndval = raster_img.nodata
        img_properties = (n_cols, n_rows, n_bands, gt, crs, ndval)
    img_ndarray = np.transpose(img_ndarray, (1, 2, 0))

    return img_ndarray, img_properties


def change_img_shape(img_ndarray_input):

    newshape = (img_ndarray_input.shape[0] * img_ndarray_input.shape[1],
                img_ndarray_input.shape[2])

    img_ndarray_output = img_ndarray_input[:,
                                           :,
                                           :img_ndarray_input.shape[2]].reshape(newshape)

    return img_ndarray_output


def map_apply_reduce_pandas(img_ndarray_input, model, shape):
    img_output_cpu = pd.DataFrame()
    img_split = np.array_split(img_ndarray_input, 10)
    for img_segment in img_split:
        img_cupy = cp.asarray(img_segment)
        img_df = cudf.DataFrame(img_cupy)

        n_rows, c_cols = img_df.shape
        prediction = model.model.predict(img_df)
        # prediction_cpu = prediction.to_pandas()
        # img_output_cpu.append(prediction_cpu)
        img_output_cpu = pd.concat(
            [img_output_cpu, prediction.to_pandas()], axis=0)

        del img_cupy, img_df, prediction

    img_output_cpu = img_output_cpu.to_numpy()
    prediction_raster = img_output_cpu.reshape(shape).astype(np.float32)
    return prediction_raster


def output_gtiff(prediction_array, img_properties, img_ndarray, current_tif, output_directory):
    n_cols, n_rows, n_bands, gt, crs, ndval = img_properties 

    prediction_array[img_ndarray[:, :, 0] == -9999.0] = np.float32(0)

    file_name_no_ext = current_tif.split('.', 1)[0]
    file_name_predicted = file_name_no_ext + '_predicted.tif'
    path = os.path.join(output_directory, file_name_predicted)
    print(file_name_predicted)
    print(path)
    with rio.open(path,
                    'w',
                    driver='GTiff',
                    height=prediction_array.shape[0],
                    width=prediction_array.shape[1],
                    count=1,
                    dtype=prediction_array.dtype,
                    crs=crs,
                    transform=gt) as prediction_raster:
        prediction_raster.write(prediction_array, 1)
