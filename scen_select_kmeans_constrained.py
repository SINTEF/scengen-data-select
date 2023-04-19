"""
scenario selection using "constrained k-means" algorithm from
https://adared.ch/constrained-k-means-implementation-in-python/
"""

import scengen_common as sgc

from third_party.constrained_kmeans.constrained_kmeans import constrained_kmeans
import pandas as pd
import numpy as np
import logging
logger = logging.getLogger(__name__)


class SelectByCKMeans(sgc.SelectorBase):
	def __init__(self, parSG: dict):
		super().__init__(parSG)
		
		# method-specific parameters from parSG (which comes from a json file)
		parKM = parSG.get('k-means', dict()) # type: dict
		if 'nmb-runs' in parKM.keys():
			self._NmbRuns = parKM['nmb-runs']
		self._MaxIter = parKM.get('max-iter', None)
	
		self._FixedPrec = 1e6  # all values are multiplied by this and rounded to an integer

	
	def run(self, df: pd.DataFrame, season = '', nScen: int = None) -> [dict, str]:
		"""
		this runs the selection method

		arguments:
		- df = data frame with the data series; its index is used to identify the selection
		- nScen = number of scenarios/sequences to select (if different from self._nmbScen)
		"""
		if nScen == None:
			nScen = self._nmbScen

		if float(pd.__version__[2:]) >= 24:
			Days = df.index.to_list()
		else:
			# to_list() was introduces in 0.24, tolist() is deprecated
			Days = df.index.tolist()

		nData = len(Days)
		lb = [nData // nScen] * nScen  # lower bound on cluster sizes
		prob = 1 / nScen

		# The return values are:
		# C – cluster centers
		# M – assignments of point to clusters
		# f – last minimum cost flow solution
		C, M, f = constrained_kmeans(df.to_numpy(), lb, self._MaxIter, fixedprec=self._FixedPrec)
		logger.debug(f"constrained_kmeans: C = {C}")
		logger.debug(f"constrained_kmeans: M = {M}")
		del f  # not used for anything

		check = 0
		resProb = dict()
		for c in range(nScen):
			cIdx = [n for n in range(nData) if M[n] == c]
			check += len(cIdx)
			selDay = -1
			minDist = np.inf
			for i in cIdx:
				nCoord = df.loc[Days[i], :].to_numpy()
				dist = np.linalg.norm(nCoord - C[c])
				if dist < minDist:
					selDay = i
					minDist = dist
			resProb[Days[selDay]] = prob
		assert check == nData, "all nodes must be assigned"
		
		# TODO: adjust constrained_kmeans() to return iteration count?
		# nIters = self._kmeans.n_iter_  # number of iterations (in the best run?)
		# maxIter = self._kmeans.get_params()['max_iter']
		# resStatus = "ok" if nIters < maxIter else "max-iter reached"
		resStatus = "ok"

		return resProb, resStatus
