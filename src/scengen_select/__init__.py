"""scengen-data-select package."""

# make the main() function from scengen_select.py available for import
from scengen_select.scripts.scengen_select import main as cli_main

# make the main selection classes available for import
# TMP: had comment out some methods, since their were causing module errors
from scengen_select.scen_select_kmeans import SelectByKMeans
#from scengen_select.scen_select_kmeans_constrained import SelectByCKMeans
#from scengen_select.scen_select_kmeans_samesize import SelectBySSKMeans
#from scengen_select.scen_select_kmedoids import SelectByKMedoids
from scengen_select.scen_select_optimize import SelectByOptimize
from scengen_select.scen_select_sampling import SelectBySampling
# NB: there are 2 versions of SelectBySamplingWD, one using pyomo and one Xpress
#     we expose the pyomo-one, since it is more generic
from scengen_select.scen_select_sampling_WD_pyomo import SelectBySamplingWD
from scengen_select.scen_select_scenred import SelectByScenRed
from scengen_select.scen_select_Wasserstein import SelectByWasserstein
