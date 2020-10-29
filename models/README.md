# Models

All model objects are abstractions for the cuml_dask or cuml -based models.

### Current functionalty includes:

##### Dask-based models
1. Model parameters updates `my_dask_model.update_params(new_parameters)`
2. Model evaluation `my_model.metrics(X_test, y_test)`

##### cuML-based models
1. Model parameters updates `my_dask_model.update_params(new_parameters)`
2. Model evaluation `my_model.metrics(X_test, y_test)`
3. Model feature importances `my_model.get_importances(X_train, y_train)`


###### To come:
1. Saving functionality
2. Dask model cross validation