
# Import system related modules

import os
import cupy as cp

# Import rapids specific modules

import cudf as df
from cuml import train_test_split



class LakeDepth():
	
	def __init__(self, random_state):
		# Set all object-specific variables
		self.predictor = ['Depth_m']
		self.cols_to_drop = ['FID', 'Date']
		self.FEATURES_PATH = os.path.join(os.path.join(os.getcwd(), 'load_dataset'), 'LakeDepth')
		print(self.FEATRUES_PATH)
		self.random_state = random_state

		#Read data into GPU as a cuDF DataFrame
		print(" - from DATA: reading csv into GPU memory")
		self.data_df = df.read_csv(self.FEATURES_PATH)
		print(" - from DATA: done reading csv into GPU memory")
		
		# Go through dataset and drop user-given columns to drop
		for col_to_drop in cols_to_drop:
			self.data_df = self.data_df.drop([col_to_drop], axis = 1)
			print(" - from DATA: dropped column:", col_to_drop)

		# Split off the labels and covariates from the dataset
		self.labels = self.data_df[self.predictor].astype(cp.float32)
		self.covariates = self.data_df.drop([self.predictor], axis = 1).astype(cp.float32)

	def split(self, test_size):
		""" Abstraction of cuml train_test_split that works with object's internal data """
		cv_train, cv_test, labels_train, labels_test = train_test_split(self.covariates,
																		self.labels,
																		test_size = test_size,
																		random_state = self.random_state)
		return cv_train, cv_test, labels_train, labels_test
