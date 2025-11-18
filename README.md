Scenario-generation by selection from historical data
=====================================================

Python implementation of scenario-generation methods described in paper __Scenario generation by selection from historical data__, published in _Computational Management Science_, volume 18, pages 411�429 (2021); [DOI: 10.1007/s10287-021-00399-4](https://doi.org/10.1007/s10287-021-00399-4).

This includes two interfaces, one for the TIMES energy-system model and one generic, using csv files as input and output.

Python setup
------------

At the moment, this repo does not provide an installable package, only two scripts together with supporting files.
All required packages are given in `requirements.txt`, installable with `pip install -r requirements.txt`.

__NB:__ To use the the optimization method, one needs to install a solver with python binding and then specify the solver as an argument to the script. Some of the python bindings are listed in `requirements.txt`, but are commented out by default.

### Problems with 'scikit-learn-extra'

As per November 2025, the 'scikit-learn-extra' package does not work with recent python and/or numpy >= 2.0.
For this reason, we require numpy 1.x and recommend python <= 3.12. (Python 3.13 did not work, at least not on linux.)
Note that we use this package only for the 'k-medoids' method - so one could install without it if one does not need k-medoids.

If the problem does not get resolved, we should consider switching to an alternative implementation, such as [kmedoids](https://pypi.org/project/kmedoids/).


Files
----

### Main scripts

- `scengen_select.py`
	- script implementing a generic interface
	- accepts command-line arguments, run with `--help` for an overview
	- in addition, all options can be specified in a JSON configuration file,
	  following the schema from `scen-gen_select.schema.json`

- `scengen_times.py`
	- script implementing interface for TIMES and Empire energy-optimization models
	- all input data can be specified in a JSON configuration file
	  following the schema from `scen-gen_times.schema.json`
	- most common options can be set using command-line arguments, run with `--help` for an overview


### Model files

- `scengen_common.py`
	- common definitions
- `scen_select_optimize.py`
	- selection using optimization (MIP), implemented in Pyomo
- `scengen_mod.py`
	- Pyomo model for use in `scen_select_optimize.py`
- `scengen_mod_equiprob.py`
	- version of the model from `scengen_mod.py` for equiprobable scenarios
- `scen_select_kmeans.py`
	- selector using _k_-means, in particular python implementation from sklearn
- `scen_select_kmeans_constrained.py`
	- selector using constrained _k_-means alg. from <https://adared.ch/constrained-k-means-implementation-in-python/>
- `scen_select_kmeans_samesize.py`
	- selector using "same-size _k_-means" algorithm from <https://elki-project.github.io/tutorial/same-size_k_means>
- `scen_select_sampling.py`
	- selector using the "iterative sampling" approach, with moment-based metric for evaluation
- `scen_select_sampling_WD.py`
	- selector using the "iterative sampling" approach, with Wasserstein-based metric for evaluation
	- computes the distance using the analytical formula from Section 3.4 of the paper
- `scen_select_Wasserstein.py`
	- selector minimizing the Wasserstein distance, using a heuristic

### Alternative implementations of the Wasserstein distance

Originally, the algorithm used an optimization model to evaluate the Wasserstein distances.
While this is no longer in use (since the analytical formula is simpler and faster), we keep the files here just in case.
The model was implemented in Pyomo and in Xpress Python libraries.

- `scen_select_sampling_WD_xpr.py`
	- variant of `scen_select_sampling_WD.py` using optimization with Xpress Python library
- `scengen_mod_Wasserstein_xpr.py`
	- Xpress model used from `scen_select_sampling_WD_xpr.py`
- `scen_select_sampling_WD_pyomo.py`
	- variant of `scen_select_sampling_WD.py` using optimization with Pyomo
- `scengen_mod_Wasserstein_pyomo.py`
	- Pyomo model used from `scen_select_sampling_WD_pyomo.py`
