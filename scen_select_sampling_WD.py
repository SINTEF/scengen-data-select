# scenario selection iterative sampling, with scenarios evaluated
# using Wasserstein distance from the data

from scengen_mod_Wasserstein import build_model
import scengen_common as sgc

import math
import pandas as pd
import numpy as np
import pyomo.environ as pyo
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
		self._write_lp_file = False

		self._aMod = build_model()  # abstract model (no data)
		self._solMng = pyo.SolverManagerFactory('serial')  # standard local solver
		self._opt = pyo.SolverFactory('xpress', solver_io='lp', is_mip=False)
	

	def sample_dist(self, scDayIdx: list) -> float:
		"""
		computes the distance of a given sample from the data
		"""
		logger.debug("solving the Wasserstein-minimizing LP")

		# update the objective function
		logger.debug(" - updating objective of the model instance")
		self._mod.obj = sum(self._Dist[d, scDayIdx[s]] * self._mod.pm[d,s] for (d,s) in self._mod.Days * self._mod.ScDays)

		if self._write_lp_file:
			logger.debug(" - writing the instance as an .lp file")
			self._mod.write('scen_select_sampling_W.lp', io_options={'symbolic_solver_labels': True})

		try:
			logger.debug(" - solving the model instance")
			res = self._solMng.solve(self._mod, opt=self._opt, tee=False)
		except Exception:
			print(f"Error during solver execution:\n - type: {sys.exc_info()[0]}\n - msg.: {sys.exc_info()[1]}")

		logger.debug(f" - solver status = {res.solver.status}")
		if self._mod.obj.expr() is None:
			print("ERROR: empty objective value -> aborting!\n")
			quit()
		logger.debug(f" - total distance = {self._mod.obj.expr()}")

		return self._mod.obj.expr()


	def run(self, df: pd.DataFrame, season = '', nScen: int = None) -> [dict, str]:
		"""
		this runs the selection method

		arguments:
		- df = data frame with the data series; must have an index called Date
		- nScen = number of scenarios/sequences to select (if different from self._nmbScen)
		"""
		if nScen == None:
			nScen = self._nmbScen

		Days = df.index.to_list()
		nData = len(Days)
		data = df.to_numpy()  # numpy array is faster to work with than dataframe!
		self._Dist = np.array([[np.linalg.norm(data[i] - data[j]) for i in range(nData)] for j in range(nData)])
		self._nData = nData  # save for later
		
		# build the common part of the data dictionary
		dDict = {
			'Days': {None: range(nData)},
			'ScDays': {None: range(nScen)},
			'Dist': {(day,sc) : self._Dist[day,sc] for sc in range(nScen) for day in range(nData)},
		}
		mDict = {None: dDict}
		self._mod = self._aMod.create_instance(mDict)

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
