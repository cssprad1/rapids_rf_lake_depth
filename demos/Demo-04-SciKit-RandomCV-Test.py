import random
from pprint import pprint
import numpy as np
import cupy as cp
from pprint import pprint

from models import custom_RF as md
from load_dataset import custom_lakedepth as data_prep
from pprint import pprint

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

if __name__ == '__main__':

    data = data_prep.LakeDepth(42)
    cv_train, cv_test, l_train, l_test = data.split(0.20)

    N_ESTIMATORS = [int(x) for x in np.linspace(start=600, stop=2000, num=10)]
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

    rf = md.cuRF(None)
    rf_random = RandomizedSearchCV(estimator=rf.model,
                                   param_distributions=random_grid,
                                   n_iter=1,
                                   cv=3,
                                   verbose=2,
                                   random_state=42)

    rf_random.fit(cv_train, l_train)
    pprint(rf_random.best_params_)

    best_random = rf_random.best_estimator_
    rf.model = best_random
    rf.get_metrics(cv_test, labels_test)
    predictions = best_random.predict(cv_test)
    mae_score = mean_absolute_error(l_test.to_pandas(), predictions.to_pandas())
    r2 = r2_score(l_test.to_pandas(), predictions.to_pandas())
    mse_score = mean_squared_error(l_test.to_pandas(), predictions.to_pandas())
    print('Mean Absolute Error: {:0.4f} meters.'.format(mae_score))
    print('Mean Squared Error: {:0.4f}'.format(mse_score))
    print('r2 score: {:0.4f}'.format(r2))
