# scenario selection iterative sampling, with scenarios evaluated
# using distance in moments and correlations

from . import scengen_common as sgc

import math
import pandas as pd
import numpy.random as rnd
import logging
logger = logging.getLogger(__name__)


class SelectBySampling(sgc.SelectorBase):
	def __init__(self, parSG: dict):
		super().__init__(parSG)
		
		# method-specific parameters from parSG (which comes from a json file)
		parS = parSG.get('sampling', dict()) # type: dict
		# if 'nmb-runs' in parKM.keys():
		# 	self._kmeans.set_params(n_init=parKM['nmb-runs'])
		# if 'nmb-threads' in parKM.keys():
		# 	self._kmeans.set_params(n_jobs=parKM['nmb-threads'])
		self._nSamples = parS.get('nmb-samples', 1000)
		self._momWeights = {
			'mean': parS.get('weight-mean', 10),
			'std': parS.get('weight-stdev', 5),
			'skew': parS.get('weight-skew', 2),
			'kurt': parS.get('weight-kurt', 1)
		}
		self._corrWeights = parS.get('weight-corr', 2)
		self._showProgress = parS.get('show-progress', True)
	

	def sample_dist(self, dfSc: pd.DataFrame, tgM, tgC) -> float:
		"""
		computes the distance of a given sample from the targets
		"""
		# moments
		scM = sgc.df_moments(dfSc)
		difM = abs(scM - tgM).mean()
		# OBS: requires that self._momWeights uses the same keys as the data frames!
		dist = sum([difM[m] * w for m, w in self._momWeights.items() if w > 0])

		#correlations
		if self._corrWeights > 0:
			scC = dfSc.corr()
			difC = abs(scC - tgC).mean().mean()
			dist += self._corrWeights * difC

		return dist


	def run(self, df: pd.DataFrame, season = '', nScen: int = None) -> [dict, str]:
		"""
		this runs the selection method

		arguments:
		- df = data frame with the data series; its index is used to identify the selection
		- nScen = number of scenarios/sequences to select (if different from self._nmbScen)
		"""
		if nScen == None:
			nScen = self._nmbScen

		# df.index.tolist() was replaced by .to_list() in pandas 0.24
		# - instead of checking for version, we simply try the new one first..
		try:
			Days = df.index.to_list()
		except:
			Days = df.index.tolist()

		dfTgM = sgc.df_moments(df)
		dfTgC = df.corr()
		
		# number of digits required for the highest sample number
		nDigits = math.floor(math.log10(self._nSamples)) + 1

		minDist = math.inf
		nFailed = 0

		for i in range(self._nSamples):
			scDays = set(rnd.choice(Days, nScen, replace=False))
			# filter df by index: https://stackoverflow.com/a/45040370/842693
			# OBS: without .copy(), this is just a read-only slice!
			dfSc = df[df.index.isin(scDays)]
			
			# NB: sample_dist may fail if all values in one series are the same,
			#     so standard deviation is zero (so higher moments and correlations are not defined)
			#   : this happens esp. with low number of scenarios
			try:
				dist = self.sample_dist(dfSc, dfTgM, dfTgC)
			except ValueError:
				nFailed += 1
				continue

			if dist < minDist:
				bestSel = scDays.copy()  # OBS: without copy, we would get a reference!
				if self._showProgress:
					print(f" - new best sample: iter = {i+1:{nDigits}d}, dist = {dist:.3f}")
				minDist = dist

		if nFailed > 0:
			w = "WARNING: " if nFailed > self._nSamples / 2 else ""
			print(f" - {w}{nFailed} out of {self._nSamples} samples discarded (zero stdev)")

		if minDist == math.inf:
			# did not find a single valid sample point
			return dict(), "failed: no valid sample"
		
		# create the output dictionary
		resProb = { d : 1/nScen for d in bestSel }

		# create the output status
		resStatus = "ok"

		return resProb, resStatus
