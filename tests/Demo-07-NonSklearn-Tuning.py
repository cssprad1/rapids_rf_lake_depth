from models.custom_RF import cuRF
from load_dataset.avaris_dataset import Avaris
from pprint import pprint
from src import randomized_cv_search as rscv

if __name__ == '__main__':
    avaris_1 = Avaris()
    results = rscv.random_cv_search(80, 3, avaris_1.covariates, avaris_1.labels)
    pprint(results)

    