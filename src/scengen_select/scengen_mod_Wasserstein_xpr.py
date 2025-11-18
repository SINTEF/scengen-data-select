import xpress as xpr

"""
	This builds the scen-gen model as using Xpress Python libraries
"""
def build_model(N: int, S: int, Dist, free_prob: bool = True, min_prob_mult = 0.1):
	m = xpr.problem(name = "Wasserstein dist. minimizer")

	pm = {(i,s): xpr.var(name=f'pi_{i}_{s}') for i in range(N) for s in range(S)}
	m.addVariable(pm)

	if free_prob:
		p = {s: xpr.var(lb=min_prob_mult * 1 / S, name=f'p_{s}') for s in range(S)}
		m.addVariable(p)
		m.addConstraint(xpr.Sum([p[s] for s in range(S)]) == 1)

	# alt. 1: using a constraint object
	c_sum_prob_out = [xpr.Sum([pm[i,s] for s in range(S)]) == 1 / N for i in range(N)]
	m.addConstraint(c_sum_prob_out)

	# alt. 2: creating the constraint directly
	if free_prob:
		m.addConstraint([xpr.Sum([pm[i,s] for i in range(N)]) == p[s] for s in range(S)])
	else:
		m.addConstraint([xpr.Sum([pm[i,s] for i in range(N)]) == 1 / S for s in range(S)])

	m.setObjective(xpr.Sum([Dist[i,s] * pm[i,s] for s in range(S) for i in range(N)]), sense=xpr.minimize)

	# finished
	return m


def main():
	import numpy as np
	from timeit import default_timer as timer
	import sys

	N = 10
	I = 3
	S = 2

	Days = range(N)
	Scens = range(S)

	np.random.seed(0)
	Data = np.random.rand(N, I)
	Data[0:5] *= 10  # first 5 values are 10 times bigger

	# print(DistA.shape)
	# print(DistA)

	for freeProb in {False, True}:
		print("\n-------------------------------------------------")
		print(f"MODEL WITH {'FREE' if freeProb else 'FIXED'} PROBABILITIES\n")

		# creating the model .. bad selection
		ScIdx = [2, 4]
		DistA = np.array([[np.linalg.norm(Data[i] - Data[ScIdx[s]]) for s in Scens] for i in Days])

		print(" - creating the model")
		m = build_model(N, S, DistA, freeProb)

		print(" - writing the instance as an .lp file")
		m.write(f'scengen_mod_W_xpr_test_1_{freeProb}', 'l')
		
		m.setControl({
			'outputlog': 0
		})

		print(" - solving the model instance")
		tStart = timer()
		try:
			m.solve()
		except Exception:
			print(f"Error during solver execution:\n - type: {sys.exc_info()[0]}\n - msg.: {sys.exc_info()[1]}")
		print(f" - model instance solved in {timer() - tStart:.0f} s")
		print(f" - solver status = {m.getProbStatus()} ('{m.getProbStatusString()}')")
		print(f" - total distance = {m.getObjVal()}")


		# ----------------------------
		print('\nRepeating with different (better) scenario selection:')
		ScIdx = [2, 6]
		DistA = np.array([[np.linalg.norm(Data[i] - Data[ScIdx[s]]) for s in Scens] for i in Days])

		obj_coeffs = [DistA[i,s] for i in range(N) for s in range(S)]
		m.chgobj(range(20), obj_coeffs)
		m.write(f'scengen_mod_W_xpr_test_2_{freeProb}', 'l')

		print(" - solving the model instance")
		tStart = timer()
		try:
			m.solve()
		except Exception:
			print(f"Error during solver execution:\n - type: {sys.exc_info()[0]}\n - msg.: {sys.exc_info()[1]}")
		print(f" - model instance solved in {timer() - tStart:.0f} s")
		print(f" - solver status = {m.getProbStatus()} ('{m.getProbStatusString()}')")
		print(f" - total distance = {m.getObjVal()}")


if __name__ == "__main__":
	# execute only if run as a script
	main()
