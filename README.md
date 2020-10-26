# Random Forest for Lake Depth

Random Forest model implemented using RAPIDS AI's cuML and cuDF which run data manipulation, reading, training, etc on GPUs available instead of host CPUs. 

This random forest model takes in a csv which contains instances of tabulated spectral data from a specific raster stack. Each of the rows has 35 data-points which are the covariates. The predictor in this instance is the lake depth. 

