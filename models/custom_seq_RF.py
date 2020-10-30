
import time

# Import RAPIDS related modules

import cuml
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

class CumlRF():

	def __init__(self, param):
		self.hyper_params = param
		#self.model = cumlDaskRF(n
		if param is None:
			self.model = cumlDaskRF()
		else:
			self.model = cumlDaskRF(n_estimators = param['N_ESTIMATORS'],
								split_algo = param['SPLIT_ALGO'],
								split_criterion = param['SPLIT_CRITERION'],
								bootstrap = param['BOOTSTRAP'],
								bootstrap_features = param['BOOTSTRAP_FEATURES'],
								rows_sample = param['ROWS_SAMPLE'],
								max_depth = param['MAX_DEPTH'],
								max_leaves = param['MAX_LEAVES'],
								max_features = param['MAX_FEATURES'],
								n_bins = param['N_BINS'],
								min_rows_per_node = param['MIN_ROWS_PER_NODE'],
								min_impurity_decrease = param['MIN_IMPURITY_DECREASE'],
								accuracy_metric = param['ACCURACY_METRIC'],
								quantile_per_tree = param['QUANTILEPT'],
								seed = param['SEED'],
								verbose = param['VERBOSE'])

	def update_model_params(self, param):
		self.hyper_params = param
		self.model = cumlDaskRF(n_estimators = param['N_ESTIMATORS'],
								split_algo = param['SPLIT_ALGO'],
								split_criterion = param['SPLIT_CRITERION'],
								bootstrap = param['BOOTSTRAP'],
								bootstrap_features = param['BOOTSTRAP_FEATURES'],
								rows_sample = param['ROWS_SAMPLE'],
								max_depth = param['MAX_DEPTH'],
								max_leaves = param['MAX_LEAVES'],
								max_features = param['MAX_FEATURES'],
								n_bins = param['N_BINS'],
								min_rows_per_node = param['MIN_ROWS_PER_NODE'],
								min_impurity_decrease = param['MIN_IMPURITY_DECREASE'],
								accuracy_metric = param['ACCURACY_METRIC'],
								quantile_per_tree = param['QUANTILEPT'],
								seed = param['SEED'],
								verbose = param['VERBOSE'])

	def train(self, covariates_train, labels_train):
		
		self.model.fit(covariates_train, labels_train)

	def get_metrics(self, covariates_test, labels_test):

		predictions = self.model.predict(covariates_test).to_array()
		mae_score = mean_absolute_error(labels_test.to_array(), predictions)
		r2 = r2_score(labels_test.to_array(), predictions)
		mse = mean_squared_error(labels_test.to_array(), predictions)
		print("Scores ------")
		print(" MAE: ", mae_score)
		print("  r2: ", r2)
		print(" MSE: ", mse)

		return mae_score, r2, mse

	def feature_importances(self, cv_train, labels_train, show = False):
		perm_imp = permutation_importance(self.model, cv_train, labels_train)
		sorted_idx = perm_imp = resilt.importances_mean.argsort()
		sorted_idx = np.flip(sorted_idx)
		importance = perm_imp.importances_mean
		for i, v in enumerate(importance[sorted_idx]):
			print('Feature: %0d, Score: %.5f' %(i,v))
		plt.figure(figsize=(20,8))
		plt.bar([x for x in range(len(importance))], importance[sorted_idx])
		plt.xticks(range(len(importance)), list(cv_train.to_pandas().columns[sorted_idx]))
		plt.title("Mean Permutation_importance")
		
		if show is True:
			plt.show()
		else:
			plt.savefig('plt_saved_.png')
