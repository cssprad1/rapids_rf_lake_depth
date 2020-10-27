
import dask_preperation as dp
from dask.distributed import Client, wait
import model as md
import data_preperation as data_prep

if __name__ == '__main__':

	# We need to define some dask things here for testing
	dask_test_0 = dp.Dask(1, 8)
	dask_test_0.set_client(Client(dask_test_0.cluster))
	# Data time
	cols_to_drop = ['FID', 'Date']
	predictor = 'Depth_m'
	data_test_0 = data_prep.Data('data/pts_merged_final.csv', predictor, cols_to_drop, 42)
	X_t, X_test, y_t, y_test = data_test_0.split(0.20)

	hyperparameters = { 
					   'N_ESTIMATORS' : 2000,
					   'SPLIT_ALGO' : 1,
					   'SPLIT_CRITERION' : 2,
					   'BOOTSTRAP' : True,
					   'BOOTSTRAP_FEATURES' : False,
					   'ROWS_SAMPLE' : 1.0,
					   'MAX_DEPTH' : 16,
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

	from pprint import pprint
	rf = md.DaskCumlRF(hyperparameters)
	pprint(rf.hyper_params)
	
	print("persist dask data")
	cv_train_dask, labels_train_dask = dask_test_0.distribute(X_t, y_t)
	cv_test_dask, labels_test_dask = dask_test_0.distribute(X_test, y_test)
	print(" Test training ")
	rf.train(cv_train_dask, labels_train_dask)

	rf.get_metrics(cv_test_dask, labels_test_dask)

	


