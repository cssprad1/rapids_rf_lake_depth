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
import joblib
import numpy as np
import matplotlib.pyplot as plt
# Import RAPIDS related modules
import cuml

from cuml.ensemble import RandomForestRegressor as clRF
from cuml.dask.common import utils as dask_utils
from dask.distributed import wait
from cuml.dask.ensemble import RandomForestRegressor as cumlDaskRF

# Calculate metrics with these modules
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
#from dask_ml.metrics import r2_score, mean_absolute_error, mean_squared_error


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
        # """Update the model parameters to your heart's desire
        # <param> model parameters to update model with"""
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
                <covariates_train> covariates (x) to train model on
                <labels_train> labels (y) to train model on
        '''
        self.model.fit(covariates_train, labels_train)
        wait(self.model.rfs)

    def get_metrics(self, covariates_test, labels_test):
        '''
        Gets the metrics based off of three common regression scores: mean_absolute_error, mean_squared_error, and r2_score
        Computes the predictions
                <covaraites_test> covariates (x) to test the model on
                <labels_test> labels (y) to test the model on
        '''
        #from cuml.metrics.regression import r2_score, mean_absolute_error, mean_squared_error
        #covariates_test = covariates_test.to_pandas()
        #labels_test = labels_test.to_pandas()
        predictions = self.model.predict(covariates_test).compute()
        predictions = predictions.to_array()
        labels_test = labels_test.to_array()
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
        '''
        Initializes a cuML RF model with the given parameters
        <param> is the parameter dictionary which assigns the entire parameter suite to the RF model. 
        <model> is the actual GPU-based model
        '''
        if param is None:
            self.model = clRF()
            self.hyper_params = self.model.get_params()
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
            self.hyper_params = self.model.get_params()

    def update_model_params(self, param):
        '''
                Update the model parameters to your heart's desire
                <param> model parameters to update model with
                '''
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
        '''
        Trains the model
                <covariates_train> covariates (x) to train model on
                <labels_train> labels (y) to train model on
        '''
        self.model.fit(covariates_train, labels_train)

    def get_score(self, covaraites_test, labels_test):
        score = self.model.score(covaraites_test, labels_test)
        return score

    def get_metrics(self, covariates_test, labels_test):
        '''
        Gets the metrics based off of three common regression scores: mean_absolute_error,
        mean_squared_error, and r2_score
        Computes the predictions
                <covaraites_test> covariates (x) to test the model on
                <labels_test> labels (y) to test the model on
        '''
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

        return predictions, mae_score, r2, mse

    def feature_importances(self, cv_train, labels_train, show=False, nth_band=0, starting_index=0):
        '''
                Computes the importances of the features of the model object
                Algorithm used: permutation_importance
                <cv_train> the permutation alg uses this to find the most important features
                <labels_train>
                '''
        cv_list = list(cv_train.to_pandas().columns)
        perm_imp = permutation_importance(self.model, cv_train, labels_train)
        sorted_idx = perm_imp.importances_mean.argsort()
        sorted_idx = np.flip(sorted_idx)
        importance = -1 * perm_imp.importances_mean
        feature_importances = [(feature, (round(importance, 5))) for
                               feature, importance in zip(cv_list, importance)]
        feature_importances = sorted(
            feature_importances, key=lambda x: x[1], reverse=True)
        [print('Variables: {:20} Importance: {}'.format(*pair))
         for pair in feature_importances]
        plt.figure(figsize=(20, 8))
        plt.bar([x for x in range(len(importance))], importance[sorted_idx])
        x_tick_list = cv_train.to_pandas().columns[sorted_idx]
        x_tick_adjusted_length = []
        for tick in x_tick_list:
            x_tick_adjusted_length.append(tick[:3])
        plt.xticks(range(len(importance)), x_tick_adjusted_length)
        plt.title("Mean Permutation_importance")
        plt.gcf().subplots_adjust(bottom=0.15)

        name = 'permutation_importance_'+str(nth_band)+'_'+str(starting_index) +\
            '_'+str(time.time())+'.png'
        if show is True:
            plt.show()
            plt.savefig(name)
        else:
            plt.savefig(name)


def save_raw_model(model, filename):
    """
    helper function to save cuML model
    <model> must be a raw cuML or cuML-Dask model
    <filename> to save in saved_models directory
    """
    filename = "".join(["models/saved_models/", filename])
    joblib.dump(model, filename)


def load_model(filename):
    """
    helper function to save cuML model
    returns a custom_RF cuRF object
    """
    filename = "".join(["models/saved_models/", filename])
    loaded_model = joblib.load(filename)
    ld_model = cuRF(None)
    ld_model.model = loaded_model
    return ld_model


def load_raw_model(filename):
    filename = "".join(["models/saved_models/", filename])
    loaded_model = joblib.load(filename)
    return loaded_model
