from models import custom_RF as cm
from load_dataset import custom_lakedepth as ld
from pprint import pprint
from src import randomized_cv_search as rscv

if __name__ == '__main__':
    data = ld.LakeDepth(42)
    cv_train, cv_test, l_train, l_test = data.split(0.20)
    results = rscv.random_cv_search(300, 3, data.covariates, data.labels)
    pprint(results)

    