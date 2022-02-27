from pyomo.environ import AbstractModel, Set, Param, Var, Constraint, Objective, Binary, NonNegativeReals, minimize

"""
	This builds the scen-gen model as an AbstractModel object

	The parameters are:
	 - EPwrAsVar: if True, E[X^k] and E[XY] are variables; otherwise, they get substituted out
	 - DistAsVar: if True, the total distance is a variable; otherwise it gets substituted out
	 - ScaleVars: if True, we scale already the powers; otherwise, we scale in the objective

	Note: parameters passed this way can be used to conditionally build variables etc.
	    : on the other hand, Pyomo's Params can only be used to create Pyomo constructs,
		  such as expressions -> cannot be used as normal Python types!
"""
def build_model(EPwrAsVar: bool, DistAsVar: bool, ScaleVars: bool) -> AbstractModel:
	m = AbstractModel()
	m.name = "ScenarioGenerator"

	# sets
	m.Days = Set()  # days to choose from
	m.Meas = Set()  # measurements in each data value
	m.Corr = Set(within=m.Meas * m.Meas)  # correlations to match

	# parameters
	m.Sc = Param()
	m.Moms = Set(initialize=[1,2,3,4])
	m.MomW = Param(m.Moms, initialize={1:10.0, 2:5.0, 3:2.0, 4:1.0})
	m.CorrW = Param(m.Corr, default=3.0)
	#
	m.Data = Param(m.Meas, m.Days)
	m.TgPwr = Param(m.Meas, m.Moms)  # target values of E[X^k]
	m.TgEXY = Param(m.Corr)          # target values og E[XY]
	#
	m.MinProbRel = Param(default=0.2)  # min. prob = MinProbRel/Sc; MinProbRel <= 1
	m.MaxProbRel = Param(default=m.Sc) # max. prob = MaxProbRel/Sc; MaxProbRel >= 1

	# variables
	m.x = Var(m.Days, within=Binary)
	m.p = Var(m.Days, within=NonNegativeReals)
	m.dist_pos = Var(m.Meas, m.Moms, within=NonNegativeReals)
	m.dist_neg = Var(m.Meas, m.Moms, within=NonNegativeReals)
	m.dist_xy_pos = Var(m.Corr, within=NonNegativeReals)
	m.dist_xy_neg = Var(m.Corr, within=NonNegativeReals)
	if EPwrAsVar:
		m.epwr = Var(m.Meas, m.Moms)
		m.covar = Var(m.Corr)
	if DistAsVar:
		m.dist = Var(within=NonNegativeReals)

	m.MinProb = m.MinProbRel / m.Sc
	m.MaxProb = m.MaxProbRel / m.Sc

	# scaling
	def exp_pwr_mult(m, meas, pwr):
		return 1 / max(abs(m.TgPwr[meas, pwr]), 1e-6)  # to avoid division by zero
	def epwr_def_mult(m, meas, pwr):
		return exp_pwr_mult(m, meas, pwr) if ScaleVars else 1
	def epwr_dist_mult(m, meas, pwr):
		return 1 if ScaleVars else exp_pwr_mult(m, meas, pwr)
	#
	def covar_mult(m, meas1, meas2):
		return 1 / max(abs(m.TgEXY[meas1, meas2]), 1e-6)  # to avoid division by zero
	def cov_def_mult(m, meas1, meas2):
		return covar_mult(m, meas1, meas2) if ScaleVars else 1
	def cov_dist_mult(m, meas1, meas2):
		return 1 if ScaleVars else covar_mult(m, meas1, meas2)

	# selection constraint
	def scen_select_rule(m):
		return sum(m.x[d] for d in m.Days) == m.Sc
	m.scen_select = Constraint(rule=scen_select_rule)

	# probabilities; using lambdas for rules
	m.min_prob = Constraint(m.Days, rule=lambda m, d: m.p[d] >= m.MinProb * m.x[d])
	m.max_prob = Constraint(m.Days, rule=lambda m, d: m.p[d] <= m.MaxProb * m.x[d])
	m.sum_prob = Constraint(m.Days, rule=lambda m: sum(m.p[d] for d in m.Days) == 1.0)

	# definition of expected powers (E[X^k])
	def exp_pwr_formula(m, meas, pwr):
		return sum(m.p[d] * m.Data[meas, d]**pwr for d in m.Days) * epwr_def_mult(m, meas, pwr)

	if EPwrAsVar:
		def epwr_def_rule(m, meas, pwr):
			return m.epwr[meas, pwr] == exp_pwr_formula(m, meas, pwr)
		m.epwr_def = Constraint(m.Meas, m.Moms, rule=epwr_def_rule)

	def exp_pwr_rule(m, meas, pwr):
		if EPwrAsVar:
			return m.epwr[meas, pwr]
		else:
			return exp_pwr_formula(m, meas, pwr)

	# distance of expected powers (E[X^k])
	def dist_abs_pos_rule(m, meas, pwr):
		return m.dist_pos[meas, pwr] >= exp_pwr_rule(m, meas, pwr) - m.TgPwr[meas, pwr] * epwr_def_mult(m, meas, pwr)
	m.dist_abs_pos = Constraint(m.Meas, m.Moms, rule=dist_abs_pos_rule)

	def dist_abs_neg_rule(m, meas, pwr):
		return m.dist_neg[meas, pwr] >= m.TgPwr[meas, pwr] * epwr_def_mult(m, meas, pwr) - exp_pwr_rule(m, meas, pwr)
	m.dist_abs_neg = Constraint(m.Meas, m.Moms, rule=dist_abs_neg_rule)

	def dist_tot_moms_rule(m, meas):
		return sum(m.MomW[pwr] * (m.dist_pos[meas, pwr] + m.dist_neg[meas, pwr]) * epwr_dist_mult(m, meas, pwr) for pwr in m.Moms)

	# distance of 'correlations' (E[XY])
	def exp_xy_formula(m, meas1, meas2):
		return sum(m.p[d] * m.Data[meas1, d] * m.Data[meas2, d] for d in m.Days) * cov_def_mult(m, meas1, meas2)

	if EPwrAsVar:
		def covar_def_rule(m, meas1, meas2):
			return m.covar[meas1, meas2] == exp_xy_formula(m, meas1, meas2)
		m.covar_def = Constraint(m.Corr, rule=covar_def_rule)

	def exp_xy_rule(m, meas1, meas2):
		if EPwrAsVar:
			return m.covar[meas1, meas2]
		else:
			return exp_xy_formula(m, meas1, meas2)

	def dist_xy_abs_pos_rule(m, meas1, meas2):
		return m.dist_xy_pos[meas1, meas2] >= exp_xy_rule(m, meas1, meas2) - m.TgEXY[meas1, meas2] * cov_def_mult(m, meas1, meas2)
	m.dist_xy_abs_pos = Constraint(m.Corr, rule=dist_xy_abs_pos_rule)

	def dist_xy_abs_neg_rule(m, meas1, meas2):
		return m.dist_xy_neg[meas1, meas2] >= m.TgEXY[meas1, meas2] * cov_def_mult(m, meas1, meas2) - exp_xy_rule(m, meas1, meas2)
	m.dist_xy_abs_neg = Constraint(m.Corr, rule=dist_xy_abs_neg_rule)

	def dist_tot_corr_rule(m, meas1, meas2):
		return m.CorrW[meas1, meas2] * (m.dist_xy_pos[meas1, meas2] + m.dist_xy_neg[meas1, meas2]) * cov_dist_mult(m, meas1, meas2)

	# total distance (using a variable or expression/rule)
	def dist_tot_rule(m):
		return sum(dist_tot_moms_rule(m, meas) for meas in m.Meas) \
			+ sum(dist_tot_corr_rule(m, meas1, meas2) for (meas1, meas2) in m.Corr)
	if DistAsVar:
		def dist_def_rule(m):
			return m.dist == dist_tot_rule(m)
		m.dist_def = Constraint(rule=dist_def_rule)
		m.obj = Objective(expr=m.dist, sense=minimize)
	else:
		m.obj = Objective(rule=dist_tot_rule, sense=minimize)
	
	# finished
	return m
