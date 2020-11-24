from pprint import pprint
import time

from models.custom_RF import cuRF
from load_dataset.aviris_dataset import Aviris
from src.k_fold_train import k_fold_train

if __name__ == '__main__':

    aviris_0 = Aviris()
    aviris_0.split(0.10)

    # Params for our fancy GPU-based RF model
    hyperparameters_0 = {'N_ESTIMATORS': 389,
                         'SPLIT_ALGO': 1,
                         'SPLIT_CRITERION': 2,
                         'BOOTSTRAP': True,
                         'BOOTSTRAP_FEATURES': False,
                         'ROWS_SAMPLE': 1.0,
                         'MAX_DEPTH': 50,
                         'MAX_LEAVES': -1,
                         'MAX_FEATURES': 'auto',
                         'N_BINS': 16,
                         'MIN_ROWS_PER_NODE': 2,
                         'MIN_IMPURITY_DECREASE': 0.0,
                         'ACCURACY_METRIC': 'mean_ae',  # 'mse' #'r2' # 'median_aw' #
                         'QUANTILEPT': False,
                         'SEED':  42,
                         'VERBOSE': False
                         }
    hyperparameters_1 = {'N_ESTIMATORS': 1431,
                         'SPLIT_ALGO': 1,
                         'SPLIT_CRITERION': 2,
                         'BOOTSTRAP': True,
                         'BOOTSTRAP_FEATURES': False,
                         'ROWS_SAMPLE': 1.0,
                         'MAX_DEPTH': 70,
                         'MAX_LEAVES': -1,
                         'MAX_FEATURES': 'auto',
                         'N_BINS': 18,
                         'MIN_ROWS_PER_NODE': 2,
                         'MIN_IMPURITY_DECREASE': 0.0,
                         'ACCURACY_METRIC': 'mean_ae',  # 'mse' #'r2' # 'median_aw' #
                         'QUANTILEPT': False,
                         'SEED':  42,
                         'VERBOSE': False
                         }
    hyperparameters_2 = {'N_ESTIMATORS': 1147,
                         'SPLIT_ALGO': 1,
                         'SPLIT_CRITERION': 2,
                         'BOOTSTRAP': True,
                         'BOOTSTRAP_FEATURES': False,
                         'ROWS_SAMPLE': 1.0,
                         'MAX_DEPTH': 70,
                         'MAX_LEAVES': -1,
                         'MAX_FEATURES': 'auto',
                         'N_BINS': 13,
                         'MIN_ROWS_PER_NODE': 2,
                         'MIN_IMPURITY_DECREASE': 0.0,
                         'ACCURACY_METRIC': 'mean_ae',  # 'mse' #'r2' # 'median_aw' #
                         'QUANTILEPT': False,
                         'SEED':  42,
                         'VERBOSE': False
                         }
    hyperparameters_3 = {'N_ESTIMATORS': 957,
                         'SPLIT_ALGO': 1,
                         'SPLIT_CRITERION': 2,
                         'BOOTSTRAP': True,
                         'BOOTSTRAP_FEATURES': False,
                         'ROWS_SAMPLE': 1.0,
                         'MAX_DEPTH': 30,
                         'MAX_LEAVES': -1,
                         'MAX_FEATURES': 'auto',
                         'N_BINS': 13,
                         'MIN_ROWS_PER_NODE': 2,
                         'MIN_IMPURITY_DECREASE': 0.0,
                         'ACCURACY_METRIC': 'mean_ae',  # 'mse' #'r2' # 'median_aw' #
                         'QUANTILEPT': False,
                         'SEED':  42,
                         'VERBOSE': False
                         }
    hyperparameters_4 = {'N_ESTIMATORS': 1052,
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

    rf_0 = cuRF(hyperparameters_0)
    rf_1 = cuRF(hyperparameters_0)
    rf_2 = cuRF(hyperparameters_0)
    rf_3 = cuRF(hyperparameters_0)
    rf_4 = cuRF(hyperparameters_0)

    k_fold_train(5, rf_0, aviris_0.covariates_train, aviris_0.labels_train)
    k_fold_train(5, rf_1, aviris_0.covariates_train, aviris_0.labels_train)
    k_fold_train(5, rf_2, aviris_0.covariates_train, aviris_0.labels_train)
    k_fold_train(5, rf_3, aviris_0.covariates_train, aviris_0.labels_train)
    k_fold_train(5, rf_4, aviris_0.covariates_train, aviris_0.labels_train)

    print("1: ",rf_0.model.score(aviris_0.covariates_test, aviris_0.labels_test))
    print("2: ",rf_1.model.score(aviris_0.covariates_test, aviris_0.labels_test))
    print("3: ",rf_2.model.score(aviris_0.covariates_test, aviris_0.labels_test))
    print("4: ",rf_3.model.score(aviris_0.covariates_test, aviris_0.labels_test))
    print("5: ",rf_4.model.score(aviris_0.covariates_test, aviris_0.labels_test))
