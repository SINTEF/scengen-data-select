# scenario selection using MIP optimization with Pyomo

import scengen_common as sgc

import pyomo.environ as pyo
import pandas as pd
import numpy as np
from timeit import default_timer as timer
import logging
logger = logging.getLogger(__name__)
from timeit import default_timer as timer


class SelectByOptimize(sgc.SelectorBase):
	def __init__(self, parSG: dict):
		super().__init__(parSG)
		
		# method-specific parameters from parSG (which comes from a json file)
		parOpt = parSG.get('optimization', dict()) # type: dict
		solver = parOpt.get('solver', 'xpress')
		self._logFileBase = parOpt.get('logfile-base', '')
		self._showOutput = parOpt.get('solver-output', False)
		self._maxTime = parOpt.get('max-time', 300)  # default is 5 minut
		solMngName = 'neos' if parOpt.get('use-neos', False) else 'serial'  # 'serial' is the default pyomo solver manager

		# min/max ranges for probability
		# - the JSON schema ensures that exactly one of the below is true!
		# - an alternative is to remove the 'oneOf' property from the schema and add
		#   some default values to the code below; this way, the more 'concrete' values
		#   would overwrite the more generic ones, and 'equiprob' would trumph everything
		self._Equiprob = False
		if parOpt.get('equiprob', False):
			self._Equiprob = True
		elif 'min-prob' in parOpt.keys():
			# TODO: adjust the model so it accepts absolute min/max as well
			assert False, "absolute min/max probabilities not implemented yet"
		elif 'min-rel-prob' in parOpt.keys():
			self._MinProbRel = parOpt['min-rel-prob']
			self._MaxProbRel = parOpt['max-rel-prob']
		elif 'prob-range-mult' in parOpt.keys():
			ProbRangeMult = parOpt['prob-range-mult']
			self._MinProbRel = 1 / ProbRangeMult
			self._MaxProbRel = ProbRangeMult
		else:
			assert False, "this should not happen with valid JSON file"

		# parameters influencing the model structure
		EPwrAsVar = parOpt.get('exp-pwr-as-var', True)  # if True, E[X^k] and E[XY] are variables; otherwise, they get substituted out
		DistAsVar = parOpt.get('dist-as-var', True)     # if True, the total distance is a variable; otherwise it gets substituted out
		ScaleVars = parOpt.get('early-scaling', True)   # if True, we scale already the powers; otherwise, we scale in the objective

		# other parameters
		nMoments = 4
		self._Moms = [i + 1 for i in range(nMoments)]
		self._SaveLpFile = parOpt.get('save-lp-file', False)
		self._NamesInLpFile = parOpt.get('names-in-lp-file', True)
	
		# create the abstract model
		if self._Equiprob:
			from scengen_mod_equiprob import build_model
		else:
			from scengen_mod import build_model
		self._m = build_model(EPwrAsVar, DistAsVar, ScaleVars)

		# initialize solver manager (so we can use NEOS)
		self._solMng = pyo.SolverManagerFactory(solMngName)

		# solver-related links:
		# - GAMS: https://pyomo.readthedocs.io/en/stable/library_reference/solvers/gams.html

		# initialize the solver
		if solver == 'xpress':
			# xpress needs 'is_mip=True' to solve it as a mip!?
			opt = pyo.SolverFactory(solver, solver_io='lp', is_mip=True)
		else:
			opt = pyo.SolverFactory(solver)

		# solver options
		MaxTimeKey = {"xpress": "maxtime", "cplex": "timelimit", "cbc": "seconds", "glpk": "tmlim", "gurobi": "TimeLimit", "gams": "resLim"}
		if solver in MaxTimeKey:
			opt.options[MaxTimeKey[solver]] = self._maxTime
		else:
			logger.warning(f"Do not know hot to set up 'max time' for solver '{solver}'!")
		#
		# allow 'solver-options-num <SOLVER>' entries, so we can keep options for multiple solvers
		solOpt = parOpt.get('solver-options-num' + ' ' + solver, dict()) # type: dict
		for so in solOpt:
			opt.options[so] = solOpt[so]
		# let the standard 'solver-options-num' overwrite the above
		solOpt = parOpt.get('solver-options-num', dict()) # type: dict
		for so in solOpt:
			opt.options[so] = solOpt[so]
		logger.debug(f"solver options:\n{opt.options}")
		self._opt = opt  # store it

	
	def run(self, df: pd.DataFrame, season = '', nScen: int = None) -> [dict, str]:
		"""
		this runs the selection method

		arguments:
		- df = data frame with the data series; its index is used to identify the selection
		- nScen = number of scenarios/sequences to select (if different from self._nmbScen)
		"""
		if nScen == None:
			nScen = self._nmbScen

		dEPwr = sgc.exp_powers(df, self._Moms)
		
		# construct data dictionary for the Pyomo model
		print(" - constructing data dictionaries for the Pyomo model")
		if float(pd.__version__[2:]) >= 24:
			Days = df.index.to_list()
			Meas = dEPwr.index.to_list()
		else:
			# to_list() was introduces in 0.24, tolist() is deprecated
			Days = df.index.tolist()
			Meas = dEPwr.index.tolist()
		nMeas = len(Meas)
		Corrs = [(Meas[i], Meas[j]) for i in range(nMeas) for j in range(i+1, nMeas)]  # all comb.
		dDict = dict()
		dDict['Days'] = {None : Days}
		dDict['Meas'] = {None : Meas}
		dDict['Corr'] = {None : Corrs}
		dDict['Sc'] = {None : nScen}
		dDict['Data'] = {(meas, day) : df[meas][day] for meas in Meas for day in Days}
		dDict['TgPwr'] = {(meas, mom) : dEPwr[mom][meas] for meas in Meas for mom in self._Moms}
		# OBS: direct computation is SLOW -> use pandas covariance matrix!
		#dDict['TgEXY'] = {(m1, m2) : (df[m1] * df[m2]).mean() for (m1, m2) in Corrs}
		nDat = len(df)
		tgCov = df.cov() * (nDat-1) / nDat  # cov() normalizes with (N-1)
		tgMean = df.mean()
		dDict['TgEXY'] = {(m1, m2) : tgCov.at[m1, m2] + tgMean[m1] * tgMean[m2] for (m1, m2) in Corrs}
		if not self._Equiprob:
			dDict['MinProbRel'] = {None: self._MinProbRel}
			dDict['MaxProbRel'] = {None: self._MaxProbRel}
		mDict = {None : dDict}

		# create model instance (concrete model) with the data
		print(" - creating the model instance")
		tStart = timer()
		mi = self._m.create_instance(mDict)
		print(f"   - creation time = {timer() - tStart:.0f} s")
		if self._SaveLpFile:
			print(" - writing the instance as an .lp file")
			if self._NamesInLpFile:
				mi.write(f'scengen_{season}.lp', io_options={'symbolic_solver_labels': True})
			else:
				mi.write(f'scengen_{season}.lp')

		# solve the problem
		print(" - solving the model instance")

		# solve
		logFile = self._logFileBase + "_" + str(season) + ".log" if len(self._logFileBase) > 0 else None
		tStart = timer()
		# note: simpler syntax for local solvers: `res = opt.solve(mi, tee=showOutput, logfile=logFile)`
		try:
			res = self._solMng.solve(mi, opt=self._opt, tee=self._showOutput, logfile=logFile)
		except Exception as e:
			logger.error(f"Error during solver execution: {e}")
			return dict(), str(e)
		print(f"   - solution time = {timer() - tStart:.0f} s")

		# notes about the `res` object:
		# - `print(res)` prints a summary
		# - contains 3 objects: `res.problem`, `res.solver`, and `res.solution`
		#   - all can be printed with `print(..)`
		#   - all are some kind of lists (so we can use `len(..)` to get the count)
		#     - `res.problem` is of type `ListContainer`, while` res.problem[0]` is `ProblemInformation`
		#     - `res.solver` is of type `ListContainer`, while` res.solver[0]` is `SolverInformation`
		#     - `res.solution` is of type `SolutionSet`, while` res.solution[0]` is `Solution`
		logger.debug(res) # only for debugging

		# reporting and checks
		resStatus = f'{res.solver.status.key}_{res.solver.termination_condition.key}'
		print(" - solver status = {}".format(res.solver.status.key))
		# solver statuses: ok, warning, error, aborted, unknown (see `[x.key for x in pyo.SolverStatus]`)
		# - different solvers use either aborted or warning when stopped by time limit -> abort only on error
		if res.solver.status == pyo.SolverStatus.error:
			return dict(), resStatus

		print(" - termination condition = {}".format(res.solver.termination_condition))
		# to get a list of statuses: ``[x.key for x in pyo.TerminationCondition]
		if res.solver.termination_condition not in [pyo.TerminationCondition.optimal,
		                                            pyo.TerminationCondition.unknown,
		                                            pyo.TerminationCondition.maxTimeLimit,
		                                            pyo.TerminationCondition.userInterrupt,
		                                            pyo.TerminationCondition.resourceInterrupt,
		                                            pyo.TerminationCondition.maxIterations]:
			return dict(), resStatus
		# `len(res.solution)` should include the number of returned solutions, but it is zero,
		# at least for Xpress -> we cannot use the test below :-(
		#print(" - res contains {} solutions".format(len(res.solution)))
		#if len(res.solution) == 0:
		#	print("ERROR: no solution in the returned solution object -> aborting!\n")
		#	continue
		if mi.obj.expr() is None:
			logger.error("ERROR: empty objective value -> aborting!\n")
			dict(), resStatus
		print(" - total distance = {}".format(mi.obj.expr()))
		
		# create the return dictionary
		if self._Equiprob:
			resProb = {d: 1 / mi.Sc for d in Days if mi.x[d].value > 0.99}
		else:
			#resProb = {d: mi.p[d].value for d in Days if mi.p[d].value > 1e-9}
			resProb = {d: mi.p[d].value for d in Days if mi.x[d].value > 0.99}

		return resProb, resStatus
