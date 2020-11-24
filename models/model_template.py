import time
import joblib
import numpy as np
import matplotlib.pyplot as plt

import cuml
# from dask.distributed import wait


class Model_Template():

    def __init__(self, parameters = None):

        self.parameters = parameters
        self.model = cuml.ensemble.RandomForestRegressor()
        if self.parameters is not None:
            self.model.parameters = self.parameters
        self._random_state = 42

    def get_parameters(self):
        return self.parameters
    
    def set_parameters(self, parameters):
        self.parameters = parameters
        self.model.parameters = parameters
    
    def get_model(self):
        return self.model

    def set_model(self, model):
        self.model = model
        self.parameters = self.model.parameters
    
    def set_random_state(self, random_state):
        self._random_state = random_state

    def train(self, covariates_train, labels_train):
        self.model.fit(covariates_train, labels_train)
    
    def get_score(self, covariates_test, labels_test):
        return self.model.score(covariates_test, labels_test)

    def get_metrics(self, covariates_test, labels_test):
        predictions = self.model.predict(covariates_test)
        metric1 = None
        return metric1

    def save(self, filename):
        filename = ''.join(["models/saved_models/", filename])
        joblib.dump(self.model, filename)

    def load(filename):
        filename = ''.join(["models/saved_models/", filename])
        self.model = joblib.load(filename)
        self.parameters = self.model.parameters
        return self