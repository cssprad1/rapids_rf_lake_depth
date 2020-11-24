import random
from pprint import pprint
import numpy as np
import time

from models.custom_RF import cuRF
from load_dataset.avaris_dataset import Avaris

if __name__ == '__main__':

    avaris_0 = Avaris()

    # Params for our fancy GPU-based RF model
    hyperparameters = {'N_ESTIMATORS' : 1150,
					   'SPLIT_ALGO' : 1,
					   'SPLIT_CRITERION' : 2,
					   'BOOTSTRAP' : True,
					   'BOOTSTRAP_FEATURES' : False,
					   'ROWS_SAMPLE' : 1.0,
					   'MAX_DEPTH' : 110,
					   'MAX_LEAVES' : -1,
					   'MAX_FEATURES' : 'sqrt',
					   'N_BINS' : 6,
					   'MIN_ROWS_PER_NODE' : 2,
					   'MIN_IMPURITY_DECREASE' : 0.0,
					   'ACCURACY_METRIC' : 'mean_ae', # 'mse' #'r2' # 'median_aw' # 
					   'QUANTILEPT' : False,
					   'SEED' :  42,
					   'VERBOSE' : False
					   }

    pprint(hyperparameters)

    rf = cuRF(hyperparameters)
    # Training time
    st = time.time()
    rf.train(avaris_0.covariates_train, avaris_0.labels_train)
    et = time.time()
    print("Time to train: ", et-st)
    
    print(rf.model.score(avaris_0.covariates_test, avaris_0.labels_test))

