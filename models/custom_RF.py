"""
Version: 10.29.20
Author: Caleb Spradlin
Email: caleb.s.spradlin@nasa.gov

Decription: Houses the abstraction classes for the Random Forest models
			Adds in functionality such as training, getting metrics, etc. 
			See individual class descriptions for more information
"""
# Import system modules
import time

# Import RAPIDS related modules
import cuml

from cuml.ensemble import RandomForestRegressor as clRF
from cuml.dask.common import utils as dask_utils
from dask.distributed import wait
from cuml.dask.ensemble import RandomForestRegressor as cumlDaskRF

# Calculate metrics with these modules
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


class DaskCumlRF():
    '''
    Class: DaskCumlRF
    Description: This is an abstraction for the dask-cuml ensemble algorithm Random Forest Regressor.
    This class also adds in functionality so code does not have to be rewritten in individual scripts,
    such as updating the model parameters, and getting model metrics (evaluations)
    '''

    def __init__(self, param):
        '''
        Initializes a cuML RF model with the given parameters
        <param> is the parameter dictionary which assigns the entire parameter suite to the RF model. 
        <model> is the actual GPU-based model
        '''
        self.hyper_params = param
        if param is None:
            self.model = cumlDaskRF()
        else:
            self.model = cumlDaskRF(n_estimators=param['N_ESTIMATORS'],
                                    split_algo=param['SPLIT_ALGO'],
                                    split_criterion=param['SPLIT_CRITERION'],
                                    bootstrap=param['BOOTSTRAP'],
                                    bootstrap_features=param['BOOTSTRAP_FEATURES'],
                                    rows_sample=param['ROWS_SAMPLE'],
                                    max_depth=param['MAX_DEPTH'],
                                    max_leaves=param['MAX_LEAVES'],
                                    max_features=param['MAX_FEATURES'],
                                    n_bins=param['N_BINS'],
                                    min_rows_per_node=param['MIN_ROWS_PER_NODE'],
                                    min_impurity_decrease=param['MIN_IMPURITY_DECREASE'],
                                    accuracy_metric=param['ACCURACY_METRIC'],
                                    quantile_per_tree=param['QUANTILEPT'],
                                    seed=param['SEED'],
                                    verbose=param['VERBOSE'])

    def update_model_params(self, param):
        self.hyper_params = param
        self.model = cumlDaskRF(n_estimators=param['N_ESTIMATORS'],
                                split_algo=param['SPLIT_ALGO'],
                                split_criterion=param['SPLIT_CRITERION'],
                                bootstrap=param['BOOTSTRAP'],
                                bootstrap_features=param['BOOTSTRAP_FEATURES'],
                                rows_sample=param['ROWS_SAMPLE'],
                                max_depth=param['MAX_DEPTH'],
                                max_leaves=param['MAX_LEAVES'],
                                max_features=param['MAX_FEATURES'],
                                n_bins=param['N_BINS'],
                                min_rows_per_node=param['MIN_ROWS_PER_NODE'],
                                min_impurity_decrease=param['MIN_IMPURITY_DECREASE'],
                                accuracy_metric=param['ACCURACY_METRIC'],
                                quantile_per_tree=param['QUANTILEPT'],
                                seed=param['SEED'],
                                verbose=param['VERBOSE'])

    def train(self, covariates_train, labels_train):
        '''
        Trains the model and waits for all parallel processes to return before returning the function
        '''
        self.model.fit(covariates_train, labels_train)
        wait(self.model.rfs)

    def get_metrics(self, covariates_test, labels_test):
        '''
        Gets the metrics based off of three common regression scores: mean_absolute_error, mean_squared_error, and r2_score
        Computes the predictions
        '''
        covariates_test = covariates_test.to_pandas()
        labels_test = labels_test.to_pandas()
        predictions = self.model.predict(covariates_test)
        predictions = predictions.to_pandas()
        mae_score = mean_absolute_error(labels_test, predictions)
        r2 = r2_score(labels_test, predictions)
        mse = mean_squared_error(labels_test, predictions)
        print("Scores ------")
        print(" MAE: ", mae_score)
        print("  r2: ", r2)
        print(" MSE: ", mse)

        return mae_score, r2, mse

    def feature_importances(self, cv_train, labels_train, show=False):
        '''
        This is merely a placeholder since the other, sequential RF does implement feature importances
        '''
        print("Feature importances not supported with Dask-based RF")


class cuRF():
    '''
    Class: cuRF
    Description: This is an abstraction for the cuML ensemble algorithm Random Forest Regressor.
    This class also adds in functionality so code does not have to be rewritten in individual scripts,
    such as updating the model parameters, and getting model metrics (evaluations)
    '''
    def __init__(self, param):
        self.hyper_params = param
        if param is None:
            self.model = clRF()
        else:
            self.model = clRF(n_estimators=param['N_ESTIMATORS'],
                              split_algo=param['SPLIT_ALGO'],
                              split_criterion=param['SPLIT_CRITERION'],
                              bootstrap=param['BOOTSTRAP'],
                              bootstrap_features=param['BOOTSTRAP_FEATURES'],
                              rows_sample=param['ROWS_SAMPLE'],
                              max_depth=param['MAX_DEPTH'],
                              max_leaves=param['MAX_LEAVES'],
                              max_features=param['MAX_FEATURES'],
                              n_bins=param['N_BINS'],
                              min_rows_per_node=param['MIN_ROWS_PER_NODE'],
                              min_impurity_decrease=param['MIN_IMPURITY_DECREASE'],
                              accuracy_metric=param['ACCURACY_METRIC'],
                              quantile_per_tree=param['QUANTILEPT'],
                              seed=param['SEED'],
                              verbose=param['VERBOSE'])

    def update_model_params(self, param):
        self.hyper_params = param
        self.model = clRF(n_estimators=param['N_ESTIMATORS'],
                          split_algo=param['SPLIT_ALGO'],
                          split_criterion=param['SPLIT_CRITERION'],
                          bootstrap=param['BOOTSTRAP'],
                          bootstrap_features=param['BOOTSTRAP_FEATURES'],
                          rows_sample=param['ROWS_SAMPLE'],
                          max_depth=param['MAX_DEPTH'],
                          max_leaves=param['MAX_LEAVES'],
                          max_features=param['MAX_FEATURES'],
                          n_bins=param['N_BINS'],
                          min_rows_per_node=param['MIN_ROWS_PER_NODE'],
                          min_impurity_decrease=param['MIN_IMPURITY_DECREASE'],
                          accuracy_metric=param['ACCURACY_METRIC'],
                          quantile_per_tree=param['QUANTILEPT'],
                          seed=param['SEED'],
                          verbose=param['VERBOSE'])

    def train(self, covariates_train, labels_train):

        self.model.fit(covariates_train, labels_train)

    def get_metrics(self, covariates_test, labels_test):

        predictions = self.model.predict(covariates_test)

        mae_score = mean_absolute_error(
            labels_test.to_pandas(), predictions.to_pandas())
        r2 = r2_score(labels_test.to_pandas(), predictions.to_pandas())
        mse = mean_squared_error(
            labels_test.to_pandas(), predictions.to_pandas())
        print("Scores ------")
        print(" MAE: ", mae_score)
        print("  r2: ", r2)
        print(" MSE: ", mse)

        return mae_score, r2, mse

    def feature_importances(self, cv_train, labels_train):
        perm_imp = permutation_importance(self.model, cv_train, labels_train)
        sorted_idx = perm_imp = resilt.importances_mean.argsort()
        sorted_idx = np.flip(sorted_idx)
        importance = perm_imp.importances_mean
        for i, v in enumerate(importance[sorted_idx]):
            print('Feature: %0d, Score: %.5f' % (i, v))
        plt.figure(figsize=(20, 8))
        plt.bar([x for x in range(len(importance))], importance[sorted_idx])
        plt.xticks(range(len(importance)), list(
            cv_train.to_pandas().columns[sorted_idx]))
        plt.title("Mean Permutation_importance")

        if show is True:
            plt.show()
        else:
            plt.savefig('plt_saved_.png')
