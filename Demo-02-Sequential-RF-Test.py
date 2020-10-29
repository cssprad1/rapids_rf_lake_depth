
from models import custom_RF as md
from load_dataset import custom_lakedepth as data_prep
from pprint import pprint
import time
if __name__ == '__main__':

	#sys.path.append('..')
    data = data_prep.LakeDepth(42)
    cv_train, cv_test, l_train, l_test = data.split(0.20)
    hyperparameters = {'N_ESTIMATORS' : 2000,
					   'SPLIT_ALGO' : 1,
					   'SPLIT_CRITERION' : 2,
					   'BOOTSTRAP' : True,
					   'BOOTSTRAP_FEATURES' : False,
					   'ROWS_SAMPLE' : 1.0,
					   'MAX_DEPTH' : 50,
					   'MAX_LEAVES' : -1,
					   'MAX_FEATURES' : 'auto',
					   'N_BINS' : 8,
					   'MIN_ROWS_PER_NODE' : 2,
					   'MIN_IMPURITY_DECREASE' : 0.0,
					   'ACCURACY_METRIC' : 'mean_ae', # 'mse' #'r2' # 'median_aw' # 
					   'QUANTILEPT' : False,
					   'SEED' :  42,
					   'VERBOSE' : False
					   }
    pprint(hyperparameters)
    rf = md.cuRF(hyperparameters)
    st = time.time()
    rf.train(cv_train, l_train)
    et = time.time()
    print("Time to train: ", et-st)
    rf.get_metrics(cv_test, l_test)
    