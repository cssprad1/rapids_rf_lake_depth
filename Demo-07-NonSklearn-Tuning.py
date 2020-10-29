import numpy as np
import random
from models import custom_RF as cm
from load_dataset import custom_lakedepth as ld
from pprint import pprint

from sklearn.model_selection import KFold
from src import randomized_cv_search as rscv
import time
if __name__ == '__main__':
    data = ld.LakeDepth(42)
    cv_train, cv_test, l_train, l_test = data.split(0.20)
    results = rscv.random_cv_search(1, 2, cv_train, l_train)
    pprint(results)
	#sys.path.append('..')
    """
    # Import the LakeDepth data with random-seed of 42
    data = ld.LakeDepth(42)
    cv_train, cv_test, l_train, l_test = data.split(0.20)
    N_ESTIMATORS = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
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

    k_fold = KFold(5)
    results = []
    # Params for our fancy GPU-based RF model
    for i in range(1):
        print(" - from RS_CV: Epoch ", i)
        
        hyperparameters = {'N_ESTIMATORS' : random.choice(N_ESTIMATORS),
					   'SPLIT_ALGO' : 1,
					   'SPLIT_CRITERION' : 2,
					   'BOOTSTRAP' : random.choice(BOOTSTRAP),
					   'BOOTSTRAP_FEATURES' : False,
					   'ROWS_SAMPLE' : 1.0,
					   'MAX_DEPTH' : random.choice(MAX_DEPTH),
					   'MAX_LEAVES' : -1,
					   'MAX_FEATURES' : random.choice(MAX_FEATURES),
					   'N_BINS' : random.choice(N_BINS),
					   'MIN_ROWS_PER_NODE' : 2,
					   'MIN_IMPURITY_DECREASE' : 0.0,
					   'ACCURACY_METRIC' :  'mse', #'r2' # 'median_aw' #'mean_ae', #
					   'QUANTILEPT' : False,
					   'SEED' :  42,
					   'VERBOSE' : False
					   }
        # Init our model with the params
        pprint(hyperparameters)
        rf = cm.cuRF(hyperparameters)
        scores = []
        st_total = time.time()

        for k, (train, test) in enumerate(k_fold.split(cv_train, l_train)):
            cv_local_train, l_local_train = cv_train.iloc[train], l_train.iloc[train]
            cv_local_test, l_local_test = cv_train.iloc[test], cv_train.iloc[test]
            st_train = time.time()
            rf.train(cv_local_train, l_local_train)
            et_train = time.time()
            print("   - from RS_CV: time to train (sec):", et_train-st_train)
            score = rf.model.score(cv_local_test, l_local_test)
            scores.append(score)
        
        et_total = time.time()
        print("   - from RS_CV: time to train and score (sec):", et_total-st_total)
        results.append({'n_estimators': rf.model.n_estimators,
                    'bootstrap': rf.model.bootstrap,
                    'max_depth': rf.model.max_depth,
                    'max_features': rf.model.max_features,
                    'n_bins': rf.model.n_bins,
                    'performance': np.mean(scores)})
        """

    