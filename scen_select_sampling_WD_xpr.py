# scenario selection iterative sampling, with scenarios evaluated
# using Wasserstein distance from the data
#
# this version uses minimization via Xpress Python library

from scengen_mod_Wasserstein_xpr import build_model
import scengen_common as sgc

import math
import pandas as pd
import numpy as np
#import xpress as xpr
import sys
import logging
logger = logging.getLogger(__name__)


class SelectBySamplingWD(sgc.SelectorBase):
	def __init__(self, parSG: dict):
		super().__init__(parSG)
		
		# method-specific parameters from parSG (which comes from a json file)
		parS = parSG.get('sampling-wd', dict()) # type: dict
		self._nSamples = parS.get('nmb-samples', 1000)
		self._showProgress = parS.get('show-progress', True)
		self._free_prob = not parS.get('equiprob', False)
		self._write_lp_file = False


	def sample_dist(self, scDayIdx: list) -> float:
		"""
		computes the distance of a given sample from the data
		"""
		logger.debug("solving the Wasserstein-minimizing LP")

		# update the objective function
		logger.debug(" - updating objective of the model instance")
		obj_coeffs = [self._Dist[i,scDayIdx[s]] for i in range(self._nData) for s in range(self._nScen)]
		self._mod.chgobj(self._pmVars, obj_coeffs)

		if self._write_lp_file:
			logger.debug(" - writing the instance as an .lp file")
			self._mod.write('scen_select_sampling_W', 'l')

		try:
			logger.debug(" - solving the model instance")
			self._mod.solve()
		except Exception:
			print(f"Error during solver execution:\n - type: {sys.exc_info()[0]}\n - msg.: {sys.exc_info()[1]}")

		logger.debug(f" - solver status = {self._mod.getProbStatus()} ('{self._mod.getProbStatusString()}')")
		logger.debug(f" - total distance = {self._mod.getObjVal()}")

		return self._mod.getObjVal()


	def run(self, df: pd.DataFrame, season = '', nScen: int = None) -> [dict, str]:
		"""
		this runs the selection method

		arguments:
		- df = data frame with the data series; its index is used to identify the selection
		- nScen = number of scenarios/sequences to select (if different from self._nmbScen)
		"""
		if nScen == None:
			nScen = self._nmbScen

		Days = df.index.to_list()
		nData = len(Days)
		data = df.to_numpy()  # numpy array is faster to work with than dataframe!
		self._Dist = np.array([[np.linalg.norm(data[i] - data[j]) for i in range(nData)] for j in range(nData)])

		# save values for later
		self._nData = nData
		self._nScen = nScen
		self._pmVars = range(nData * nScen)  # range of indices of variables 'pm'
		self._pVars = range(nData * nScen, nData * nScen + nScen) if self._free_prob else []  # indices of 'p' variables

		# need some initial values, to build the model
		Dist = np.array([[self._Dist[i,s] for s in range(nScen)] for i in range(nData)])
		self._mod = build_model(nData, nScen, Dist)
		self._mod.setControl({
			'outputlog': 0
		})

		if self._free_prob:
			assert list(self._pVars) == [self._mod.getIndexFromName(2, f'p_{s}') for s in range(nScen)], "index check"

		# number of digits required for the highest sample number
		nDigits = math.floor(math.log10(self._nSamples)) + 1

		minDist = math.inf
		nFailed = 0

		for i in range(self._nSamples):
			#scDays = set(np.random.choice(Days, nScen, replace=False))
			scDayIdx = list(np.random.choice(range(nData), nScen, replace=False))
			scDays = {Days[sc] for sc in scDayIdx}
			
			# NB: fails should not happen, but we check nevertheless
			try:
				dist = self.sample_dist(scDayIdx)
			except ValueError:
				nFailed += 1
				continue

			if dist < minDist:
				bestSel = scDays.copy()  # OBS: without copy, we would get a reference!
				bestScenProb = self._mod.getSolution(self._pVars)
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
		if self._free_prob:
			resProb = { d : bestScenProb[s] for s,d in enumerate(bestSel)}
		else:
			resProb = { d : 1/nScen for d in bestSel }

		# create the output status
		resStatus = "ok"

		return resProb, resStatus
