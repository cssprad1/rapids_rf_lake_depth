# Random Forest for Lake Depth

Random Forest model implemented using RAPIDS AI's cuML and cuDF which run data manipulation, reading, training, etc on GPUs available instead of host CPUs. 

This random forest model takes in a csv which contains instances of tabulated spectral data from a specific raster stack. Each of the rows has 35 data-points which are the covariates. The predictor in this instance is the lake depth.

Current functionality is in the notebooks located in:

For cuML/cuDf prototypes: models/

For sklearn prototypes: tests/

### Dealing with conda issues

There seems to be many issues when trying to run the Jupyter Notebooks from the preloaded rapids ml conda environment kernel. 

Please follow these steps to clone the correct conda environment:

#

```
# If you haven't already, prepare your user space for personal anaconda environments by doing the following. (From any ADAPT node)
$ mv ~/.conda $NOBACKUP
$ ln -s $NOBACKUP/.conda ~/.conda
# Clone the environment to user space so that it gets brought back to the old module. (Do this on the new gpu cluster)
gpulogin1 $ module load anaconda/3-2020.07
gpulogin1 $ conda create -n rapids-0.16 -c rapidsai -c nvidia -c conda-forge \
    -c defaults rapids=0.16 python=3.7 cudatoolkit=10.2
gpulogin1 $ conda activate rapids-0.16
# Make sure ipykernel is installed so that you can use it with JH.
gpulogin1 $ conda install ipykernel
gpulogin1 $ conda install -c conda-forge dask-ml
# We are currently having some issues with our xcat images on the GPU cluster and user anaconda installs. For your own sanity run:
gpulogin1 $ chmod -R 0700 ~/.conda/envs/rapids-0.16
```
After following these steps, log onto your JH intance and you should see a JH kernel named "myRapids".
