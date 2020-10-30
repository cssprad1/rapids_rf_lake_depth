import random
import time
import numpy as np
# Import all modules for RF (RAPIDS, DASK, etc)
from load_dataset import custom_lakedepth as ld
from pprint import pprint
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from cuml.dask.common import utils as dask_utils
from dask.distributed import Client, wait
from dask_cuda import LocalCUDACluster
import dask_cudf
from cuml.dask.ensemble import RandomForestRegressor as cumlDaskRF

if __name__ == '__main__':
    cluster = LocalCUDACluster(threads_per_worker=1)
    c = Client(cluster)
    workers = c.has_what().keys()
    n_workers = len(workers)
    n_streams = 8 # Performance optimization
    data = ld.LakeDepth(42)
    covariates_train, covariates_test, labels_train, labels_test = data.split(0.20)
    covariates_test_pd = covariates_test.to_pandas()
    labels_test_pd = labels_test.to_pandas()
    k_fold = KFold(5)
    results = []
    TEST_SIZE = 0.20
    RANDOM_STATE = 42
    n_partitions = n_workers
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
    for i in range(1):
        print("  - from Random Search: Epoch #", i)

        n_estimators=random.choice(N_ESTIMATORS)
        split_algo=SPLIT_ALGO
        split_criterion=SPLIT_CRITERION
        bootstrap=random.choice(BOOTSTRAP)
        bootstrap_features=BOOTSTRAP_FEATURES
        rows_sample=ROWS_SAMPLE
        max_depth=random.choice(MAX_DEPTH)
        max_leaves=MAX_LEAVES
        max_features=random.choice(MAX_FEATURES)
        n_bins=random.choice(N_BINS)
        min_rows_per_node=MIN_ROWS_PER_NODE
        min_impurity_decrease=MIN_IMPURITY_DECREASE
        accuracy_metric=ACCURACY_METRIC
        quantile_per_tree=QUANTILEPT
        seed=SEED
        verbose=VERBOSE

        depth_rf_model = cumlDaskRF(n_estimators=n_estimators,
                          split_algo=split_algo,
                          split_criterion=split_criterion,
                          bootstrap=bootstrap,
                          bootstrap_features=bootstrap_features,
                          rows_sample=rows_sample,
                          max_depth=max_depth,
                          max_leaves=max_leaves,
                          max_features=max_features,
                          n_bins=n_bins,
                          min_rows_per_node=min_rows_per_node,
                          min_impurity_decrease=min_impurity_decrease,
                          accuracy_metric=accuracy_metric,
                          quantile_per_tree=quantile_per_tree,
                          seed=seed,
                          verbose=verbose)
        scores = []
        st = time.time()
        for k, (train, test) in enumerate(k_fold.split(covariates_train, labels_train)):
            
            X_dask_cudf = dask_cudf.from_cudf(covariates_train.iloc[train], npartitions=n_workers)
            y_dask_cudf = dask_cudf.from_cudf(labels_train.iloc[train], npartitions=n_workers)
            X_dask_cudf, y_dask_cudf = \
                dask_utils.persist_across_workers(c, [X_dask_cudf, y_dask_cudf], workers=workers)
            depth_rf_model.fit(X_dask_cudf, y_dask_cudf)
            wait(depth_rf_model.rfs)

            del X_dask_cudf, y_dask_cudf
            
            X_dask_cudf_test = dask_cudf.from_cudf(covariates_train.iloc[test], npartitions=n_workers)
            y_dask_cudf_test = dask_cudf.from_cudf(covariates_train.iloc[test], npartitions=n_workers)
            

            predictions = depth_rf_model.predict(X_dask_cudf_test).compute()
            predictions = predictions.to_array()
            score = mean_absolute_error(labels_train.iloc[test].to_array(), predictions)
            scores.append(score)
            et = time.time()

            del X_dask_cudf_test, y_dask_cudf_test

            print("   -time to train (sec): ", et-st)
        
        del depth_rf_model
        
        results.append({'n_estimators': n_estimators,
                    'bootstrap': bootstrap,
                    'max_depth': max_depth,
                    'max_features': max_features,
                    'n_bins': n_bins,
                    'performance': np.mean(scores)})

    pprint(results)