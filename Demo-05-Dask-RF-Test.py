
from src import dask_preperation as dp
from dask.distributed import Client, wait
from models import custom_RF as md
from load_dataset import custom_lakedepth as data_prep
from pprint import pprint
import time
import dask_cudf
if __name__ == '__main__':

	#sys.path.append('..')
    
    dask = dp.Dask(1, 8)
    c = Client(dask.cluster)
    dask.set_client(c)
    print(c)
    cols = ['FID', 'Date']
    pred = 'Depth_m'
    data = data_prep.LakeDepth(42)
    cv_train, cv_test, l_train, l_test = data.split(0.20)
    hyperparameters = {	   'N_ESTIMATORS' : 200,
					   'SPLIT_ALGO' : 1,
					   'SPLIT_CRITERION' : 2,
					   'BOOTSTRAP' : True,
					   'BOOTSTRAP_FEATURES' : False,
					   'ROWS_SAMPLE' : 1.0,
					   'MAX_DEPTH' : 8,
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
    rf = md.DaskCumlRF(hyperparameters)
    cv_dt, l_dt = dask.distribute(cv_train, l_train)
    st = time.time()
    rf.train(cv_dt, l_dt)
    et = time.time()
    print("Time to train: ", et-st)
    del cv_dt, l_dt
    X_cudf = dask_cudf.from_cudf(cv_test, npartitions=dask.n_workers)
    wait(X_cudf)
    rf.get_metrics(X_cudf, l_test)
    