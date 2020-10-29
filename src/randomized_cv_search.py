# Import all system modules

import random
import time

# Import all modules for RF (RAPIDS, DASK, etc)
from src import dask_preperation as dp
from dask.distributed import Client, wait
from load_dataset import custom_lakedepth as ld
from pprint import pprint

from cuml.dask.common import utils as dask_utils
from dask.distributed import wait
from cuml.dask.ensemble import RandomForestRefressor as clRF

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def random_cv_search(N_SEARCH, covariates, labels):
    
	k_fold = KFold(5)
	results = []
	TEST_SIZE = 0.20
	RANDOM_STATE = 42
    # Introduce lists to the hyperparameters we want to 

	N_ESTIMATORS = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
	SPLIT_ALGO = 1
	SPLIT_CRITERION = 2
	BOOTSTRAP = [True, False]
	BOOTSTRAP_FEATURES = False
	ROWS_SAMPLE = 1.0
	MAX_DEPTH = [int(x) for x in np.linspace(10, 110, num = 11)]
	MAX_LEAVES = -1
	MAX_FEATURES = ['auto', 'sqrt']
	N_BINS = [int(x) for x in np.linspace(start = 5, stop = 20, num = 10)]
	MIN_ROWS_PER_NODE = 2
	MIN_IMPURITY_DECREASE = 0.0
	ACCURACY_METRIC = 'mean_ae' # 'mse' #'r2' # 'median_aw' # 
	QUANTILEPT = False
	SEED = 42
	VERBOSE = False
	
	random_grid = {'n_estimators' : N_ESTIMATORS,
                  'max_depth' : MAX_DEPTH,
                  'bootstrap' : BOOTSTRAP,
                  'max_features': MAX_FEATURES,
                  'n_bins' : N_BINS}
	
	pprint(random_grid)
	
	for i in range(N_SEARCH):
		print("  - from Random Search: Epoch #", i)
		depth_rf_model = clRF(n_estimators = random.choice(N_ESTIMATORS), 
                            split_algo = SPLIT_ALGO, 
                            split_criterion = SPLIT_CRITERION, 
                            bootstrap = random.choice(BOOTSTRAP),
                            bootstrap_features = BOOTSTRAP_FEATURES, 
                            rows_sample = ROWS_SAMPLE,
                            max_depth = random.choice(MAX_DEPTH), 
                            max_leaves = MAX_LEAVES, 
                            max_features = random.choice(MAX_FEATURES),
                            n_bins = random.choice(N_BINS),
                            min_rows_per_node = MIN_ROWS_PER_NODE,
                            min_impurity_decrease = MIN_IMPURITY_DECREASE,
                            accuracy_metric = ACCURACY_METRIC,
                            quantile_per_tree = QUANTILEPT,
                            seed = SEED,
                            verbose = VERBOSE)

		scores = []
		st = time.time()
            
		for k, (train, test) in enumerate(k_fold.split(covariates, labels)):
			cv_train, l_train = dask.distribute(covariates.iloc[train],
					labels.iloc[train])
			cv_test, l_test = dask.distribute(covariates.iloc[train],
					labels.iloc[train])
            
			depth_rf_model.fit(cv_train, l_train)
			wait(depth_rf_model.rfs) 
			pred = depth_rf_model.predict(cv_test).compute().to_array()
			score = mean_absolute_error(covariates.to_array(), pred)
                #score = depth_rf_model.score(covariates.iloc[test],
				#labels.iloc[test])
			scores.append(score)
                
			et = time.time()
			print("   -time to train (sec): ", et-st)
			
			results.append({'n_estimators':depth_rf_model.n_estimators,
                            'bootstrap':depth_rf_model.bootstrap,
                            'max_depth':depth_rf_model.max_depth,
                            'max_features':depth_rf_model.max_features,
                            'n_bins':depth_rf_model.n_bins,
                            'performance':np.mean(scores)})

		results.sort(key=lambda x : -x['performance'])
		return results

if __name__ == '__main__':
	dask_0 = dp.Dask(1,8)
	c = Client(dask_0.cluster)
	dask_0.set_client
	data = ld.LakeDepth(42)
	cv_train, cv_test, l_train, l_test = data.split(0.20)
	list_out = random_cv_search(5, cv_train, cv_test)
	pprint(list_out)

