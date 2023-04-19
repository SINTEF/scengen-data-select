# scenario selection using the "fast forward selection" version of
# the scenario-reduction method
# Algorithm 2.4 from Heitsch & Römisch (2015)

import scengen_common as sgc

import math
import pandas as pd
import numpy as np
import logging
logger = logging.getLogger(__name__)


class SelectByScenRed(sgc.SelectorBase):
	"""
	"""
	
	def __init__(self, parSG: dict = {}):
		super().__init__(parSG)
		
		# method-specific parameters from parSG (which comes from a json file)
		parS = parSG.get('Wasserstein', dict()) # type: dict
		self._free_prob = not parS.get('equiprob', False)
		
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
		self._iDist = [[np.linalg.norm(self._data[i] - self._data[j]) for j in range(self._nData)] for i in range(self._nData)]


	def fast_forward_selection(self, nScen: int):
		# initialization
		J = set(range(self._nData))  # set for fast element deletion
		# NB: c = self._iDist[i][j].copy() does shallow copy -> make from scratch
		c = [[self._iDist[i][j] for j in J] for i in J]
		u_sc = [None] * nScen  # list of the selected scenarios
		i = 0

		while True:
			# OBS: c[k][u] should be multiplied by prob[k], but since we assume
			#      equiprobable data and do not need the actual distances,
			#      we can drop it
			z = [(u, sum([c[k][u] for k in J])) for u in J]
			# z contains pairs (u, dist) -> find min by the second element:
			z_min = min(z, key=lambda el: el[1])
			u_sc[i] = z_min[0]

			if i == nScen - 1:
				# have all 
				break
			
			J.remove(u_sc[i])
			# adjust distances
			for k in J:
				c_k_u = c[k][u_sc[i]]
				for u in J:
					c[k][u] = min(c[k][u], c_k_u)
			i += 1
		
		return u_sc


	def run(self, df: pd.DataFrame, season = '', nScen: int = None) -> [dict, str]:
		"""
		this runs the selection method

		arguments:
		- df = data frame with the data series; its index is used to identify the selection
		- nScen = number of scenarios/sequences to select (if different from self._nmbScen)
		"""
		if nScen == None:
			nScen = self._nmbScen

		self.setup_data(df)

		# this runs the scenario-reduction alg.
		scen_idx = self.fast_forward_selection(nScen)
		scen_sel = [self._index[i] for i in scen_idx]

		# construct the output dictionary: selected days -> probabilities
		if self._free_prob:
			dProb = 1.0 / self._nData
			# again using the analytical solution from Theorem 2:
			scProb = [0] * nScen
			for i in range(self._nData):
				sc = np.argmin([self._iDist[i][s] for s in scen_idx])
				scProb[sc] += dProb

			resProb = { d : scProb[s] for s,d in enumerate(scen_sel)}
		else:
			scProb : 1.0 / nScen
			resProb = { d : scProb for d in scen_sel }

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

	nScen = 4

	SR = SelectByScenRed({'nmb-scen': nScen})
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
