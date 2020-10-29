
from models import custom_RF as cm
from load_dataset import custom_lakedepth as ld
from pprint import pprint
import time
if __name__ == '__main__':

	#sys.path.append('..')
    
    # Import the LakeDepth data with random-seed of 42
    data = ld.LakeDepth(42)
    cv_train, cv_test, l_train, l_test = data.split(0.20)

    # Params for our fancy GPU-based RF model
    hyperparameters = {'N_ESTIMATORS' : 200,
					   'SPLIT_ALGO' : 1,
					   'SPLIT_CRITERION' : 2,
					   'BOOTSTRAP' : True,
					   'BOOTSTRAP_FEATURES' : False,
					   'ROWS_SAMPLE' : 1.0,
					   'MAX_DEPTH' : 20,
					   'MAX_LEAVES' : -1,
					   'MAX_FEATURES' : 'auto',
					   'N_BINS' : 16,
					   'MIN_ROWS_PER_NODE' : 2,
					   'MIN_IMPURITY_DECREASE' : 0.0,
					   'ACCURACY_METRIC' : 'mean_ae', # 'mse' #'r2' # 'median_aw' # 
					   'QUANTILEPT' : False,
					   'SEED' :  42,
					   'VERBOSE' : False
					   }
    pprint(hyperparameters)

    # Init our model with the params
    rf = cm.cuRF(hyperparameters)
    
    # Training time
    st = time.time()
    rf.train(cv_train, l_train)
    et = time.time()
    print("Time to train: ", et-st)
    
    rf.get_metrics(cv_test, l_test)
    rf.feature_importances(cv_train, l_train, show = False)
    