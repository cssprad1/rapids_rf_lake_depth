# Import all system modules
import time

from pprint import pprint
from sklearn.model_selection import KFold

from models import custom_RF as cm


def k_fold_train(folds, model, nth_band, start_index, cv_train, l_train):

    k_fold = KFold(folds)
    scores = []
    st_total = time.time()

    for k, (train, test) in enumerate(k_fold.split(cv_train, l_train)):
        print("   - from RS_CV: Fold #:", k)
        cv_local_train, l_local_train = cv_train.iloc[train], l_train.iloc[train]
        cv_local_test, l_local_test = cv_train.iloc[test], cv_train.iloc[test]
        st_train = time.time()
        model.train(cv_local_train, l_local_train)
        et_train = time.time()
        print("   - from RS_CV: time to train (sec):", et_train-st_train)
        score_local = model.get_score(cv_local_test, l_local_test)
        print("   - from RS_CV: Score:", score_local)
        scores.append(score_local)

    et_total = time.time()
    print(' - from k_fold: time to train and eval:', et_total - st_total)
    cm.save_raw_model(
        model, 'k_fold_' + str(nth_band) + '_' + str(start_index) + '_estimators')

    return model
