# scenario selection using k-means clustering algorithm from sklearn

import scengen_common as sgc

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import logging
logger = logging.getLogger(__name__)


class SelectByKMeans(sgc.SelectorBase):
	def __init__(self, parSG: dict):
		super().__init__(parSG)
		
		self._kmeans = KMeans(n_clusters=self._nmbScen)

		# method-specific parameters from parSG (which comes from a json file)
		parKM = parSG.get('k-means', dict()) # type: dict
		self._equiprob = parKM.get('equiprob', False)
		# k-means alg. controls: setting only values that are given (leaving rest to default)
		if 'nmb-runs' in parKM.keys():
			self._kmeans.set_params(n_init=parKM['nmb-runs'])
		if 'nmb-threads' in parKM.keys():
			self._kmeans.set_params(n_jobs=parKM['nmb-threads'])
		if 'max-iter' in parKM.keys():
			self._kmeans.set_params(max_iter=parKM['max-iter'])
	
	
	def run(self, df: pd.DataFrame, season = '', nScen: int = None) -> [dict, str]:
		"""
		this runs the selection method

		arguments:
		- df = data frame with the data series; must have an index called Date
		- nScen = number of scenarios/sequences to select (if different from self._nmbScen)
		"""
		if nScen == None:
			nScen = self._nmbScen

		if float(pd.__version__[2:]) >= 24:
			Days = df.index.to_list()
		else:
			# to_list() was introduces in 0.24, tolist() is deprecated
			Days = df.index.tolist()

		#data = df.copy()
		#del data['Days']

		dist = self._kmeans.fit_transform(df)
		scens = self._kmeans.labels_

		# for each cluster, get index of the point/date closest to its centroid
		dIdx = dist.argmin(axis=0)

		# create the output dictionary of probabilities
		if self._equiprob:
			resProb = { Days[dIdx[sc]]: 1/nScen for sc in range(nScen) }
		else:
			# assign probability based on cluster size
			nData = len(df)
			sList = list(scens)  # need list for .count()
			prob = [sList.count(sc) / nData for sc in range(nScen)]
			
			# create the output dictionary
			resProb = { Days[dIdx[sc]]: prob[sc] for sc in range(nScen) }

		# create the output status
		nIters = self._kmeans.n_iter_  # number of iterations (in the best run?)
		maxIter = self._kmeans.get_params()['max_iter']
		resStatus = "ok" if nIters < maxIter else "max-iter reached"

		return resProb, resStatus
