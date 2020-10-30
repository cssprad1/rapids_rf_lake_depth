# Import all system modules

import random
import time

from pprint import pprint
import numpy as np
from models import custom_RF as cm
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def random_cv_search(N_SEARCH, folds, cv_train, l_train):

    print(" - from RSCV: N_SEARCH =", N_SEARCH)
    print(" - from RSCV: folds =", folds)
    print(" - from RSCV: total fits:", int(folds * N_SEARCH))
    k_fold = KFold(folds)
    results = []
    # Introduce lists to the hyperparameters we want to
    N_ESTIMATORS = [int(x) for x in np.linspace(start=200, stop=2000, num=20)]
    SPLIT_ALGO = 1
    SPLIT_CRITERION = 2
    BOOTSTRAP = [True, False]
    BOOTSTRAP_FEATURES = False
    ROWS_SAMPLE = 1.0
    MAX_DEPTH = [int(x) for x in np.linspace(10, 110, num=11)]
    MAX_LEAVES = -1
    MAX_FEATURES = ['auto', 'sqrt']
    N_BINS = [int(x) for x in np.linspace(start=5, stop=20, num=10)]
    MIN_ROWS_PER_NODE = 2
    MIN_IMPURITY_DECREASE = 0.0
    ACCURACY_METRIC = 'mean_ae'  # 'mse' #'r2' # 'median_aw' #
    QUANTILEPT = False
    SEED = 42
    VERBOSE = False

    random_grid = {'n_estimators': N_ESTIMATORS,
                   'max_depth': MAX_DEPTH,
                   'bootstrap': BOOTSTRAP,
                   'max_features': MAX_FEATURES,
                   'n_bins': N_BINS}

    pprint(random_grid)

    for i in range(N_SEARCH):
        # print(" - from RS_CV: Epoch ", i)
        hyperparameters = {'N_ESTIMATORS': random.choice(N_ESTIMATORS),
                       'SPLIT_ALGO': 1,
                       'SPLIT_CRITERION': 2,
                       'BOOTSTRAP': random.choice(BOOTSTRAP),
                       'BOOTSTRAP_FEATURES': False,
                       'ROWS_SAMPLE': 1.0,
                       'MAX_DEPTH': random.choice(MAX_DEPTH),
                       'MAX_LEAVES': -1,
                       'MAX_FEATURES': random.choice(MAX_FEATURES),
                       'N_BINS': random.choice(N_BINS),
                       'MIN_ROWS_PER_NODE': 2,
                       'MIN_IMPURITY_DECREASE': 0.0,
                       'ACCURACY_METRIC':  'mean_ae', #'mse',  # 'r2' # 'median_aw' #
                       'QUANTILEPT': False,
                       'SEED':  42,
                       'VERBOSE': False
                       }
    	# Init our model with the params
        pprint(hyperparameters)
        rf = cm.cuRF(hyperparameters)
        scores = []
        st_total = time.time()
        
        for k, (train, test) in enumerate(k_fold.split(cv_train, l_train)):
            print("   - from RS_CV: Fold #:", k)
            cv_local_train, l_local_train = cv_train.iloc[train], l_train.iloc[train]
            cv_local_test, l_local_test = cv_train.iloc[test], cv_train.iloc[test]
            st_train = time.time()
            rf.train(cv_local_train, l_local_train)
            et_train = time.time()
            print("   - from RS_CV: time to train (sec):", et_train-st_train)
            score_local = rf.get_score(cv_local_test, l_local_test)
            print("   - from RS_CV: Score:", score_local)
            scores.append(score_local)

        et_total = time.time()
        print("   - from RS_CV: time to train and score (sec):", et_total-st_total)
        results.append({'n_estimators': rf.model.n_estimators,
                    'bootstrap': rf.model.bootstrap,
                    'max_depth': rf.model.max_depth,
                    'max_features': rf.model.max_features,
                    'n_bins': rf.model.n_bins,
                    'performance': np.mean(scores)})
    results.sort(key=lambda x: -x['performance'])
    return results
