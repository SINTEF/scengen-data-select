from pyomo.environ import AbstractModel, Set, Param, Var, Constraint, Objective, Binary, NonNegativeReals, minimize

"""
	This builds the scen-gen model as an Pyomo AbstractModel object

	Note: parameters passed to this functions can be used to conditionally build variables etc.
	    : on the other hand, Pyomo's Params can only be used to create Pyomo constructs,
	      such as expressions -> cannot be used as normal Python types!
"""
def build_model() -> AbstractModel:
	m = AbstractModel()
	m.name = "ScenarioGenerator"

	# sets
	m.Days = Set()  # days to choose from
	m.ScDays = Set(within=m.Days)  # selected days, i.e., scenarios
	# m.Dist can be defined either on Days * ScDays, or Days * days
	# - the first option sends only data required, but changing for each scenario set
	# - the second opption is independent on ScDays
	m.Dist = Param(m.Days * m.Days)  # distance between days and scenarios

	# parameters, with initializers for equi-probable case
	def day_prob_init(m, day):
		return 1 / len(m.Days)
	m.Prob = Param(m.Days, initialize=day_prob_init)       # probabilities od data points
	def scen_prob_init(m, scen):
		return 1 / len(m.ScDays)
	m.ScProb = Param(m.ScDays, initialize=scen_prob_init)  # scenario probabilities

	# variables
	m.pm = Var(m.Days, m.ScDays, within=NonNegativeReals)  # moved probabilities

	# constraints
	def sum_prob_out_rule(m, day):
		return sum(m.pm[day, sc] for sc in m.ScDays) == m.Prob[day]
	m.sum_prob_out = Constraint(m.Days, rule=sum_prob_out_rule)

	def sum_prob_in_rule(m, sc):
		return sum(m.pm[day, sc] for day in m.Days) == m.ScProb[sc]
	m.sum_prob_in = Constraint(m.ScDays, rule=sum_prob_in_rule)

	# objective
	def W_dist_rule(m):
		return sum(m.Dist[day, sc] * m.pm[day, sc] for (day, sc) in m.Days * m.ScDays)
	m.obj = Objective(rule=W_dist_rule, sense=minimize)
	
	# finished
	return m


def main():
	import numpy as np
	from timeit import default_timer as timer
	import pyomo.environ as pyo
	import sys

	N = 10
	I = 3
	Scens = [2, 4]

	Days = range(N)
	
	Data = np.random.rand(N, I)
	Data[0:5] *= 10  # first 5 values are 10 times bigger
	Dist = np.array([[np.linalg.norm(Data[i] - Data[j]) for i in Days] for j in Days])

	dDict = {
		'Days': {None: Days},
		'Dist': {(day,sc) : Dist[day,sc] for sc in Days for day in Days},
		'ScDays': {None: Scens}
	}
	mDict = {None: dDict}

	print(" - creating the abstract model")
	m = build_model()
	print(" - creating the model instance")
	mi = m.create_instance(mDict)
	#mi.pprint()
	print(" - writing the instance as an .lp file")
	mi.write('scengen_mod_W-eval_test.lp', io_options={'symbolic_solver_labels': True})

	print(" - initializing the solver")
	solMng = pyo.SolverManagerFactory('serial')  # standard local solver
	opt = pyo.SolverFactory('xpress', solver_io='lp', is_mip=False)

	logFile = "scengen_mod_W-eval_test.log"
	print(" - solving the model instance")
	tStart = timer()
	# note: simpler syntax for local solvers: `res = opt.solve(mi, tee=showOutput, logfile=logFile)`
	try:
		res = solMng.solve(mi, opt=opt, tee=False, logfile=logFile)
	except Exception:
		print(f"Error during solver execution:\n - type: {sys.exc_info()[0]}\n - msg.: {sys.exc_info()[1]}")
	
	print(f" - model instance solved in {timer() - tStart:.0f} s")
	print(f" - solver status = {res.solver.status}")
	if mi.obj.expr() is None:
		print("ERROR: empty objective value -> aborting!\n")
		quit()
	print(f" - total distance = {mi.obj.expr()}")
	print(f" - pm = {[(day, sc, mi.pm[day,sc].value) for sc in Scens for day in Days]}")
	#mi.pprint()

	# ----------------------------
	print('\nRepeating with differnt (better) scenario selection:')
	Scens = [2, 6]
	Dist = np.array([[np.linalg.norm(Data[i] - Data[j]) for i in Days] for j in Days])
	mDict[None]['ScDays'] = {None: Scens}

	mi = m.create_instance(mDict)
	tStart = timer()
	# note: simpler syntax for local solvers: `res = opt.solve(mi, tee=showOutput, logfile=logFile)`
	try:
		res = solMng.solve(mi, opt=opt, tee=False)
	except Exception:
		print(f"Error during solver execution:\n - type: {sys.exc_info()[0]}\n - msg.: {sys.exc_info()[1]}")
	
	print(f" - model instance solved in {timer() - tStart:.0f} s")
	print(f" - solver status = {res.solver.status}")
	if mi.obj.expr() is None:
		print("ERROR: empty objective value -> aborting!\n")
		quit()
	print(f" - total distance = {mi.obj.expr()}")
	print(f" - pm = {[(day, sc, mi.pm[day,sc].value) for sc in Scens for day in Days]}")

	
	# ----------------------------
	print('\nAlternative approach building the model only once:')
	
	# for this, we need keep fixed indexing!
	Scens = range(2)
	ScIdx = [2, 4]

	dDict = {
		'Days': {None: Days},
		'Dist': {(day,sc) : Dist[day,ScIdx[sc]] for sc in Scens for day in Days},
		'ScDays': {None: Scens}
	}
	mDict = {None: dDict}
	
	mi = m.create_instance(mDict)
	tStart = timer()
	# note: simpler syntax for local solvers: `res = opt.solve(mi, tee=showOutput, logfile=logFile)`
	try:
		res = solMng.solve(mi, opt=opt, tee=False)
	except Exception:
		print(f"Error during solver execution:\n - type: {sys.exc_info()[0]}\n - msg.: {sys.exc_info()[1]}")
	
	print(f" - model instance solved in {timer() - tStart:.0f} s")
	print(f" - solver status = {res.solver.status}")
	if mi.obj.expr() is None:
		print("ERROR: empty objective value -> aborting!\n")
		quit()
	print(f" - total distance = {mi.obj.expr()}")

	# ----------------------------
	print('\nAnd now we update the objective function directly:')
	ScIdx = [2, 6]
	mi.obj = sum(Dist[d, ScIdx[s]] * mi.pm[d,s] for (d,s) in mi.Days * mi.ScDays)

	tStart = timer()
	# note: simpler syntax for local solvers: `res = opt.solve(mi, tee=showOutput, logfile=logFile)`
	try:
		res = solMng.solve(mi, opt=opt, tee=False)
	except Exception:
		print(f"Error during solver execution:\n - type: {sys.exc_info()[0]}\n - msg.: {sys.exc_info()[1]}")
	
	print(f" - model instance solved in {timer() - tStart:.0f} s")
	print(f" - solver status = {res.solver.status}")
	if mi.obj.expr() is None:
		print("ERROR: empty objective value -> aborting!\n")
		quit()
	print(f" - total distance = {mi.obj.expr()}")


if __name__ == "__main__":
	# execute only if run as a script
	main()
