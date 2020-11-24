import os
import cudf
import cupy as cp
from cuml import train_test_split


class DataSet():
    """
    This is a dataset object template class.
    This template can be used to copy and fill in
    what is needed to form a clean implementation
    of a dataset as an object.
    """
    INIT_TEST_SIZE = 42

    def __init__(self):

        self.predictor = ''

        self.columns_to_drop_from_dataset = []

        self.DATASET_PATH = os.path.join(os.getcwd(), 'data_directory',
                                         'project_name_directory',
                                         'example.csv')

        self._RANDOM_STATE = 42

        print(" - from DATA: reading csv into GPU memory")
        self.full_dataset = cudf.read_csv(self.DATESET_PATH)
        print(" - from DATA: done reading csv into GPU memory")

        for column in self.columns_to_drop_from_dataset:
            self.full_dataset = self.full_dataset.drop([column], axis=1)
            print(" - from DATA: dropped column ", column)

        self.labels = self.full_dataset[self.predictor].astype(cp.float32)

        self.covariates = self.full_dataset.drop(
            [self.predictor], axis=1).astype(cp.float32)

        self.covariates_train, self.covariates_test, \
            self.labels_train, self.labels_test = train_test_split(self.covariates,
                                                                   self.labels,
                                                                   test_size=INIT_TEST_SIZE,
                                                                   shuffle=True,
                                                                   random_state=self._RANDOM_STATE)

    # ---------------------------------------------
    # Getters and setters for all public variables
    # ---------------------------------------------

    def set_predictor_string(self, predictor_name):

        self.predictor = predictor_name

    def get_predictor_string(self):

        return self.predictor

    def set_cols_to_drop_from_dataset(self, columns_to_drop_list):

        self.columns_to_drop_from_dataset.append(columns_to_drop_list)

    def get_cols_to_drop_from_dataset(self):

        return self.columns_to_drop_from_dataset

    def set_dateset_path(self, dataset_path):

        self.DATASET_PATH

    def get_dataset_path(self):

        return self.DATASET_PATH

    # ---------------------------------------------
    # split()
    # ---------------------------------------------
    def split(self, split_ratio):
        """ A function to obfuscate train_test_split.
        In addition to splitting the data, the split data
        can be accessed through the datset object's public
        variables """
        self.covariates_train, self.covariates_test, \
            self.labels_train, self.labels_test = train_test_split(self.covariates,
                                                                   self.labels,
                                                                   test_size=split_ratio,
                                                                   shuffle=True,
                                                                   random_state=self._RANDOM_STATE)

        return self.covariates_train, self.covariates_test, self.labels_train, self.labels_test
