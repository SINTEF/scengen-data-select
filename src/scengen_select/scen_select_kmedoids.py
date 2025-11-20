"""
scenario selection using k-medoids clustering algorithm from sklearn-extra

info about the method: https://scikit-learn-extra.readthedocs.io/en/latest/modules/cluster.html#k-medoids
API documentation: https://scikit-learn-extra.readthedocs.io/en/latest/generated/sklearn_extra.cluster.KMedoids.html

TODO: replace it by k-Medoids from https://python-kmedoids.readthedocs.io - should be faster!
"""

from . import scengen_common as sgc

import pandas as pd
import numpy as np
from sklearn_extra.cluster import KMedoids
import logging
logger = logging.getLogger(__name__)


class SelectByKMedoids(sgc.SelectorBase):
	def __init__(self, parSG: dict):
		super().__init__(parSG)
		
		self._kmedoids = KMedoids(n_clusters=self._nmbScen)

		# method-specific parameters from parSG (which comes from a json file)
		parKM = parSG.get('k-medoids', dict()) # type: dict
		self._equiprob = parKM.get('equiprob', False)
		# k-medoids alg. controls: setting only values that are given (leaving rest to default)
		if 'max-iter' in parKM.keys():
			self._kmedoids.set_params(max_iter=parKM['max-iter'])
		if 'method' in parKM.keys():
			self._kmedoids.set_params(method=parKM['method'])  # values: {‘alternate’, ‘pam’}, default: ‘alternate’
		if 'init' in parKM.keys():
			self._kmedoids.set_params(init=parKM['init'])  # values: {‘random’, ‘heuristic’, ‘k-medoids++’, ‘build’}, optional, default: ‘build’
	
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

		if float(pd.__version__[2:]) >= 24:
			Days = df.index.to_list()
		else:
			# to_list() was introduces in 0.24, tolist() is deprecated
			Days = df.index.tolist()

		dist = self._kmedoids.fit(df)
		scens = self._kmedoids.labels_  # assignment of data points to clusters
		dIdx = self._kmedoids.medoid_indices_  # indices of the medoids

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
		nIters = self._kmedoids.n_iter_  # number of iterations (in the best run?)
		maxIter = self._kmedoids.get_params()['max_iter']
		resStatus = "ok" if nIters < maxIter else "max-iter reached"

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

	nScen = 4

	# using the 'pam' methods - slower, but should give better results
	SR = SelectByKMedoids({'nmb-scen': nScen, 'k-medoids': {'method': 'pam'}})
	prob, _ = SR.run(df)

	import matplotlib.pyplot as plt
	plt.scatter(df['col_1'], df['col_2'], marker='.')

	dfsc = df[df.index.isin(prob.keys())].copy()
	dfsc['prob'] = dfsc.index.map(prob)
	#print(f"{dfsc}")
	plt.scatter(dfsc['col_1'], dfsc['col_2'], marker='*', s=36 * nScen*dfsc['prob'])
	print("\nClose the chart to finish.")
	plt.show()


if __name__ == "__main__":
	""" This is executed when run from the command line """
	main()
