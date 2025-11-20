# scenario selection using the Wassertein-metric based algorithm from
# Pflug & Pichler (2015)

from . import scengen_common as sgc

import math
import pandas as pd
import numpy as np
import logging
logger = logging.getLogger(__name__)


class SelectByWasserstein(sgc.SelectorBase):
	"""
	Selector using a modified Wasserstein-metric-based heuristic from Pflug & Pichler (2015)

	- in particular, we implement a variant of Algorithm 2 from the paper, plus Algorithm 1
	  for generating initial solutions
	- however, Algorithm 1 is very slow, so there is an option to turn it off and
	  start from a random sample instead - or, more precisely, from Voronoi clusters
	  implied by the sample
	- the internal notation follows the paper, with Z for scenarios (list of data points)
	  and VZ for clusters (list of set of points)
	
	Selected data-points (and clusters thereof) are internally described using their
	row numbers in the input dataframe; the dataframe index is used only on output.
	"""
	
	def __init__(self, parSG: dict = {}):
		super().__init__(parSG)
		
		# method-specific parameters from parSG (which comes from a json file)
		#parS = parSG.get('Wasserstein', dict()) # type: dict
		
		self._useAlg1 = False            # whether to use alg_1 for init, or just use a random sample
		self._initUnitClusters = False   # whether alg_1 starts from unit clusters (as in the paper)
		self._initNumClustersMult = 1.1  # number of init clusters in alg_1, as a multiple of nScen
		self._initNumRandStart = 10      # number of random samples to test (if not using alg_1)

		self._iDist = None  # will include the matrix of distances between all indices (data points)
		self._index = None  # will include the index set of the data set
	

	def setup_data(self, df: pd.DataFrame):
		"""
		initial setup, using a data frame
		- inside the class, data points are identified by their index in the data frame
		- this means that also the clusters are given in terms of indices
		"""

		# store the input data, in slightly different format
		self._index = df.index
		self._data = df.to_numpy()  # numpy array is faster to work with than dataframe!
		self._nData = len(df.index)

		# self._iDist includes a 2D array of distances between data points
		# - stored, as we typically need the same distance several times
		# - OBS: unless we use the version of alg_1 with unit-size clusters,
		#        we don't need all the pairs!
		# -> create an array of None and fill it on demand - see data_dist()
		#self._iDist = [[np.linalg.norm(self._data[i] - self._data[j]) for j in range(self._nData)] for i in range(self._nData)]
		self._iDist = [[None] * self._nData for _ in range(self._nData)]
		#self._nDistComp = 0    # number of dist. computations
		#self._nDistCached = 0  # number of times we used a cached dist.


	def data_dist(self, i, j) -> float:
		"""
		return the pre-computed distance of two data points, given by their indeces
		"""
		if i == j:
			return 0.0
		if self._iDist[i][j] is not None:
			#self._nDistCached += 1
			return self._iDist[i][j]
		elif self._iDist[j][i] is not None:
			#self._nDistCached += 1
			return self._iDist[j][i]
		else:
			dist = np.linalg.norm(self._data[i] - self._data[j])
			self._iDist[i][j] = dist
			#self._nDistComp += 1
			return dist


	def cluster_dist(self, CI1: set, CI2: set) -> float:
		"""
		CI1, CI2 are sets of indices
		"""
		dist = max([self.data_dist(i1, i2) for i1 in CI1 for i2 in CI2])
		return dist


	def alg_1(self, nScen: int) -> list:
		"""
		hierarchical clustering to get a starting point
		"""
		# clustering given as a set of sets of indices,
		
		if self._initUnitClusters:
			# initialized to each index being in its own cluster
			Clusters = [{i} for i in range(self._nData)]
		else:
			nC0 = int(self._initNumClustersMult * nScen)  # init number of clusters
			Z0 = list(np.random.choice(range(self._nData), nC0, replace=False))
			Clusters, _ = self.Voronoi_clusters(Z0)

		while len(Clusters) > nScen:
			# find and merge a cluster-pair with the smallest distance
			nC = len(Clusters)
			minDist = math.inf
			for i in range(nC):
				for j in range(i+1, nC):
					d = self.cluster_dist(Clusters[i], Clusters[j])
					if d < minDist:
						logger.debug(f"merging cluster {i} and {j}")
						bestPair = (i, j)
			
			# now, merge the two clusters into one and delete the other one
			i, j = bestPair
			Clusters[i].update(Clusters[j])
			# - we want to delete Clusters[j], but since deleting from the middle
			#   of a list is expensive, we move it to the end first...
			if j < nC - 1:
				Clusters[j], Clusters[nC-1] = Clusters[nC-1], Clusters[j]
			Clusters.pop()
		
		return Clusters


	def _cluster_center(self, VZi: set) -> (int, float):
		"""
		find a "centre" of one cluster VZi, given by a set of indices

		returns index of the centre point, plus the WS-distance
		"""
		# for each value, calculate its distance to all the others
		# OBS: quadratic in the cluster size!
		d = {i: sum([self.data_dist(i, j) for j in VZi]) for i in VZi}
		iMin, dMin = min(d.items(), key=lambda x: x[1])
		logger.debug(f"cluster {VZi} has a centre at point {iMin}")

		return iMin, dMin


	def cluster_centers(self, VZ: list) -> list:
		"""
		find centres of all clusters in VZ
		"""
		# use only the first return value (ignore the distance)
		return [self._cluster_center(VZi)[0] for VZi in VZ]


	def Voronoi_clusters(self, Z: list) -> (list, float):
		"""
		find Voronoi clusters for a set of point indices

		returns the cluster plus the WS-distance
		"""
		s = len(Z)
		VZ = [set() for i in range(s)]
		dist = 0.0
		for n in range(self._nData):
			# for each data point, find the closest scenario point (Z[i])
			# and add it to its corresponding cluster
			d = [self.data_dist(i, n) for i in Z]
			# enumerate creates a list of [(i, d[i])], then we take its min
			# based on d[i] and return the whole pair
			iMin, dMin = min(enumerate(d), key=lambda x: x[1])
			VZ[iMin].add(n)
			dist += dMin
		
		return VZ, dist

	
	def ws_dist(self, Z: list, VZ: list = []) -> float:
		"""
		calculate the WS-distance (discretization error) of a "point set" Z
		"""
		s = len(Z)
		if len(VZ) > 0:
			assert len(VZ) == s, "consistency check"
			dist = sum([sum([self.data_dist(Z[i], j) for j in VZ[i]]) for i in range(s)])
		else:
			_, dist = self.Voronoi_clusters(Z)
		return dist


	def alg_2(self, Z0: list, VZ0 = [], DZ0 = -1):
		"""
		Algorithm 2 from Pflug & Pichler (2015)
		"""
		s = len(Z0)
		Z = Z0.copy()
		if len(VZ0) > 0:
			assert len(VZ0) == s, "consistency check"
			VZ = VZ0.copy()
			if DZ0 > 0:
				DZ = DZ0
			else:
				DZ = self.ws_dist(Z, VZ)
		else:
			VZ, DZ = self.Voronoi_clusters(Z)

		while True:
			logger.info(f"dist = {DZ:5.2f}")
			Z_new = self.cluster_centers(VZ)
			VZ_new, DZ_new = self.Voronoi_clusters(Z_new)
			if DZ_new >= DZ:
				# no improvement -> stop and use the previous values
				break
			else:
				# we have a better solution
				Z = Z_new.copy()
				VZ = VZ_new.copy()
				DZ = DZ_new
		
		return Z, VZ, DZ


	def run(self, df: pd.DataFrame, season = '', nScen: int|None = None) -> tuple[dict, str]:
		"""
		this runs the selection method

		arguments:
		- df = data frame with the data series; its index is used to identify the selection
		- nScen = number of scenarios/sequences to select (if different from self._nmbScen)
		"""
		nScen = nScen or self._nmbScen
		if nScen == None:
			raise ValueError("Number of scenarios to select not specified!")

		self.setup_data(df)

		if self._useAlg1:
			VZ0 = self.alg_1(nScen)
			assert len(VZ0) == nScen, "consistency check"
			Z0 = self.cluster_centers(VZ0)
			DZ0 = -1  # unknown
		else:
			DZ0 = math.inf
			for i in range(self._initNumRandStart):
				Z = list(np.random.choice(range(self._nData), nScen, replace=False))
				VZ, DZ = self.Voronoi_clusters(Z)
				if DZ < DZ0:
					logger.debug(f"init iter {i}: new best sample, dist = {DZ}")
					Z0 = Z.copy()
					VZ0 = VZ.copy()
					DZ0 = DZ

		Z, VZ, _ = self.alg_2(Z0, VZ0, DZ0)
		assert  len(Z) == nScen, "consistency check"
		assert len(VZ) == nScen, "consistency check"

		logger.debug(f"scenarios: {Z}")
		logger.info(f"cluster sizes: {[len(c) for c in VZ]}")
		#logger.debug(f"computed distances: {self._nDistComp}")
		#logger.debug(f"  cached distances: {self._nDistCached}")

		# construct the output dictionary: selected days -> probabilities
		resProb = {self._index[Z[i]]: len(VZ[i]) / len(df) for i in range(nScen)}

		# create the output status
		resStatus = "ok"

		return resProb, resStatus


# -----------------------------------------------------------------------------

def main():
	"""
	simple test of the method, with visualization of the results
	"""
	logging.basicConfig(format="%(levelname)s: %(message)s", level = logging.INFO)
	#np.random.seed(10)

	# dummy data
	N = 50  # number of data points
	D =  2  # dimension
	data = np.random.rand(N, D)
	index = [f"row_{i+1}" for i in range(N)]
	columns = [f"col_{j+1}" for j in range(D)]
	df = pd.DataFrame(data, columns=columns, index=index)

	nScen = 5

	WS = SelectByWasserstein({'nmb-scen': nScen})
	WS.setup_data(df)

	VZ0 = WS.alg_1(nScen)
	Z0 = WS.cluster_centers(VZ0)

	Z, VZ, D = WS.alg_2(Z0, VZ0)
	assert  len(Z) == nScen, "consistency check"
	assert len(VZ) == nScen, "consistency check"

	print()
	print(f"scenarios: {Z}")
	print(f"clusters: {VZ}")
	print(f"cluster sizes: {[len(c) for c in VZ]}")
	#print()
	#print(f"computed distances: {WS._nDistComp}")
	#print(f"  cached distances: {WS._nDistCached}")

	# add a cluster column to the data frame
	cDict = dict()  # dictionary data-point index -> scenario/cluster
	for ci in range(nScen):
		for c in VZ[ci]:
			cDict[c] = ci
	df['cluster'] = [cDict[i] for i in range(WS._nData)]

	import matplotlib.pyplot as plt
	plt.scatter(df['col_1'], df['col_2'], c=df['cluster'], marker='.')
	selIdx = [df.index[i] for i in Z]  # selected indices
	dfc = df[df.index.isin(selIdx)]  # only the cluster centers
	plt.scatter(dfc['col_1'], dfc['col_2'], c=dfc['cluster'], marker='*')
	print("\nClose the chart to finish.")
	plt.show()


if __name__ == "__main__":
	""" This is executed when run from the command line """
	main()
