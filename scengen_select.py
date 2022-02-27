# Scenario generation by selection from historical data
# - algorithm from https://doi.org/10.1007/s10287-021-00399-4
# - this is a generic version, with simple matrix-based input/output

import scengen_common as sgc

import os
import sys
from contextlib import redirect_stdout
import pandas as pd
import numpy as np
from pathlib import Path
import json
from timeit import default_timer as timer
import argparse
import shutil

import logging
# this sets the default values for all created loggers
# we can change the level of individual loggers using logger.setLevel()
logging.basicConfig(format="%(levelname)s:%(name)s: %(message)s", level = logging.INFO)
logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)

# with log-level INFO or higher, we get the following line in the output:
# > INFO:numexpr.utils: NumExpr defaulting to X threads.
# this sets the level to that particular logger to 'ERROR', to avoid the message
logging.getLogger('numexpr.utils').setLevel(logging.ERROR)


def read_config(configFile: str) -> dict:
	"""
	read, parse, and validate the config file
	"""

	# parse the config file
	try:
		with open(configFile, 'r') as f:
			params = json.load(f)
	except IOError as err:
		print("Error: could not open the configuration file {}:\n - {}".format(configFile, err))
		exit(1)
	except Exception as err:
		print("Error parsing the configuration file {}:\n - {}".format(configFile, err))

	# if possible validate the config file against its schema
	schemaFile = params.get('$schema', '')
	if Path(schemaFile).is_file():
		# the schema file exist -> validate the input file
		try:
			with open(schemaFile, 'r') as f:
				schema = json.load(f)
		except Exception as err:
			print("Error parsing the json schema file {}:\n - {}".format(configFile, err))
			exit(1)
		import jsonschema
		try:
			jsonschema.validate(params, schema)
		except jsonschema.exceptions.ValidationError as err:
			print("Error: the configuration file {} does not validate against the schema from {}:\n - {}\n - {}".format(\
				configFile, schemaFile, err.message, err.path))
			exit(2)

	return params


def get_data(params: dict) -> pd.DataFrame:
	"""
	this reads the data and create a dataframe with require format
	params: dictionary containing the JSON input file
	output: pandas dataframe with the following structure:
	        - index column called 'Date', of type datetime
	        - one column per date series; column name must be a string
	"""

	parInput = params['input']  # shortcut; type: dict
	dataFile = parInput.get('filename')
	indexCol = parInput.get('index-col', None)
	colSep = parInput.get('column-sep', ',')
	# allow 'tab' as a synonym for '\t'
	if colSep == 'tab':
		colSep = '\t'  # the actual tab character ('	') works as well

	print("Reading the input file.")

	df = pd.read_csv(dataFile, sep=colSep, index_col=indexCol)
	return df


def main():

	# COMMAND-LINE ARGUMENTS
	# adding a custom help formatter, to allow for avoid line splits: https://stackoverflow.com/a/52606755/842693
	termWidth = shutil.get_terminal_size()[0]
	helpFormatter = lambda prog: argparse.HelpFormatter(prog, max_help_position=40, width=termWidth)
	parser = argparse.ArgumentParser(description='Scenario generation by selection from historical data', formatter_class=helpFormatter)

	# arguments
	parser.add_argument("-c", "--config", help="name of the configuration JSON file")
	parser.add_argument("-d", "--data", help="input data file")
	parser.add_argument("--index-col", help="name of the index column")
	parser.add_argument("--col-sep", help="column separator in the input data file")
	parser.add_argument("-s", "--nmb-scen", type=int, help="number of scenarios to generate")
	parser.add_argument("-m", "--method", choices=['optimization', 'k-means', 'sampling', 'Wasserstein', 'sampling-WD'], metavar="SELECTOR", help="selector method")
	parser.add_argument("-o", "--output", help="output file name, without extension")
	parser.add_argument("-T", "--max-time", type=int, help="time limit for the optimization method [s]")
	# using two flags (--equiprob and --freeprob) for the same value (with oposite meaning)
	# - 'store_true' and 'store_false' action generate default with the oposite value, making it impossible
	#   to detect whether the a flag was given or not -> using the generic 'store_const' instead
	parser.add_argument("-e", "--equiprob", action='store_const', const=True, dest='equiprob', default=None, help="generate equiprobable scenarios")
	parser.add_argument("--free-prob", action='store_const', const=False, dest='equiprob', help="generate scenarios with free probabilities")
	# same for --write-prob and --no-prob .. want to detect if neither was given
	parser.add_argument("-p", "--write-prob", action='store_const', const=True, dest='writeprob', default=None, help="include probabilities in the output file")
	parser.add_argument("--no-prob", action='store_const', const=False, dest='writeprob', help="do not include probabilities in the output file")
	#
	parser.add_argument("--n-samples", type=int, help="number of samples for the sampling method")
	parser.add_argument("--k-means-var", help="variant of the k-means method to use")

	# parsing
	args = parser.parse_args()

	configFile = args.config

	if configFile is not None:
		params = read_config(configFile)
	else:
		# create the required structure
		# - also add default values required to run
		params = dict()
		params['scen-gen'] = dict()
		for m in {'optimization', 'k-means', 'sampling', 'Wasserstein','sampling-WD'}:
			params['scen-gen'][m.lower()] = dict()
		params['scen-gen']['optimization']['prob-range-mult'] = np.sqrt(10)
		params['input'] = dict()
		params['output'] = dict()

	# overwrite config-file data with command-line parameters (if given)
	if args.nmb_scen is not None:
		params['scen-gen']['nmb-scen'] = args.nmb_scen
	if args.method is not None:
		params['scen-gen']['selector'] = args.method
	if args.data is not None:
		params['input']['filename'] = args.data
	if args.col_sep is not None:
		params['input']['column-sep'] = args.col_sep
	if args.index_col is not None:
		params['input']['index-col'] = args.index_col
	if args.output is not None:
		params['output']['filename-base'] = args.output
	# method-specific parameters
	parM = params['scen-gen'][args.method.lower()]  # shortcut
	if args.equiprob is not None:
		parM['equiprob'] = args.equiprob  # used by several methods -> add to all
	if args.method == 'optimization':
		if args.max_time is not None:
			parM['max-time'] = args.max_time
	elif args.method in {'sampling', 'sampling-WD'}:
		if args.n_samples is not None:
			parM['nmb-samples'] = args.n_samples
	elif args.method == 'k-means':
		if args.k_means_var is not None:
			params['scen-gen']['k-means']['variant'] = args.k_means_var

	# parameters for this script (not passed to the scen-gen methods)
	# equiProb can be given in a config file for the 'optimization' method
	if args.method == 'optimization':
		equiProb = params['scen-gen']['optimization'].get('equiprob', False)
	else:
		equiProb = False
	if args.equiprob is not None:
		equiProb = args.equiprob  # command-line overrides the default
	# output probabilities if equiProb is False, unless overriden
	writeProb = args.writeprob if args.writeprob is not None else (not equiProb)

	# read the input data
	df = get_data(params)

	# process the rest of the configuration file
	parSG = params['scen-gen'] # type: dict
	nScen = parSG['nmb-scen']

	selectorType = parSG.get('selector', 'optimization')
	print(f"Initializing selector object of type '{selectorType}'.")
	if selectorType == 'optimization':
		if 'optimization' not in parSG:
			# add default values
			parSG['optimization'] = {
				'prob-range-mult': np.sqrt(10)
			}
		from scen_select_optimize import SelectByOptimize
		selector = SelectByOptimize(parSG)
	elif selectorType == 'k-means':
		kMeansVariant = 'standard'
		if 'k-means' in parSG:
			kMeansVariant = parSG['k-means'].get('variant', 'standard')
		if kMeansVariant == 'standard':
			from scen_select_kmeans import SelectByKMeans
			selector = SelectByKMeans(parSG)
		elif kMeansVariant == 'constrained':
			from scen_select_kmeans_constrained import SelectByCKMeans
			selector = SelectByCKMeans(parSG)
		elif kMeansVariant == 'same-size':
			from scen_select_kmeans_samesize import SelectBySSKMeans
			selector = SelectBySSKMeans(parSG)
		else:
			assert False, f"unsupported k-means variant '{kMeansVariant}'"
	elif selectorType == 'sampling':
		from scen_select_sampling import SelectBySampling
		selector = SelectBySampling(parSG)
	elif selectorType == 'Wasserstein':
		from scen_select_Wasserstein import SelectByWasserstein
		selector = SelectByWasserstein(parSG)
	elif selectorType == 'sampling-WD':
		from scen_select_sampling_WD_xpr import SelectBySamplingWD
		selector = SelectBySamplingWD(parSG)
	else:
		assert False, f"unsupported selector type '{selectorType}'"

	parOut = params['output'] # type: dict
	outFileBase = parOut['filename-base']

	# calling the run() method of the selector
	tStart = timer()
	[resProb, resStatus] = selector.run(df)
	if len(resProb) == 0:
		# no results!
		print(f" - selection failed, status = {resStatus}")
		exit(1)
	print(f" - selection finished in {timer() - tStart:.1f} s")
	print(f" - returned status: {resStatus}")

	if len(resProb) != nScen:
		print(" - ERROR: wrong number of scenarios returned -> aborting!")
		print(f"nScen = {nScen}")
		print(f"len(resProb) = {len(resProb)}")
		print(f"resProb = {resProb}")
		exit(1)

	# filter df by index: https://stackoverflow.com/a/45040370/842693
	# - note: in case of multiindex: https://stackoverflow.com/a/25225009/842693
	# - filtering creates a read-only slice -> need to create a copy!
	dSel = df[df.index.isin(resProb.keys())].copy()
	if len(dSel) != nScen:
		print(" - ERROR: wrong number of non-zero probabilities returned -> aborting!")
		exit(1)
	
	# add the column of probabilities, if required
	if writeProb:
		if equiProb:
			# ignore the output probabilities and use 1/nScen instead
			dSel.insert(loc=0, column='prob', value=1/nScen)
		else:
			# resProb is a {key: value} dictionary -> insert it as pandas Series
			dSel.insert(loc=0, column='prob', value=pd.Series(resProb))
			assert np.isclose(dSel['prob'].sum(), 1.0), "probabilities must sum up to 1!"

	outFile = outFileBase + ".csv"
	print("Saving results to file " + outFile + '.')
	dSel.to_csv(outFile, float_format="%g", index=False)

	print("Done.")


if __name__ == "__main__":
	""" This is executed when run from the command line """
	main()
