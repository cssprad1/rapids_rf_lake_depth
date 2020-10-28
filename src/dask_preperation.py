# Import system modules

import os

# Import dask modules

from cuml.dask.common import utils as dask_utils
from dask.distributed import Client, wait
from dask_cuda import LocalCUDACluster
import dask_cudf


class Dask():
	def __init__(self, threads_per_worker = 1, n_streams = 8):
		self.threads_per_worker = threads_per_worker
		self.n_streams = n_streams
		self.client = None
		self.workers = None
		self.n_workers = None
		self.cluster = LocalCUDACluster(self.threads_per_worker)

		
	def set_client(self, client_input):
		if self.cluster is not None:
			self.client = client_input
			self.workers = self.client.has_what().keys()
			self.n_workers = len(self.workers)
		else:
			print("Cannot pass client to object before cluster is initialized, please  creat cluster with .init_cluster() function")
	
	def distribute(self, covariates, labels):
		n_partitions = self.n_workers
		covariates_dask = dask_cudf.from_cudf(covariates, npartitions=n_partitions)
		labels_dask = dask_cudf.from_cudf(labels, npartitions=n_partitions)

		# Persist to chache the data in active memory across cluster
		covariates_dask, labels_dask = \
				dask_utils.persist_across_workers(self.client, [covariates_dask, labels_dask], 
						workers = self.workers)

		return covariates_dask, labels_dask
	
	def set_threads_per_worker(self, tpw_input):
		self.threads_per_worker = tpw_input
	
	def set_n_streams(self, ns_input):
		self.n_streams = ns_input


if __name__ == '__main__':

	# We need to define some dask things here for testing
	print("Working from main in dask_prep module")
	dask_test_0 = Dask(1, 8)
	dask_test_0.set_client(Client(dask_test_0.cluster))
	print(dask_test_0.client)

