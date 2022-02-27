# Scenario generation for TIMES and EMPIRE

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
	dateTimeCol = parInput.get('datetime_col', 'Date')
	regionCol = parInput.get('region_col', '')
	serCols = parInput.get('series_cols')
	missingStr = parInput.get('missing_val')
	#
	parScenGen = params['scen-gen']  # shortcut; type: dict
	scenLen = parScenGen.get('scen-length')

	print("Reading the input file.")

	if len(serCols) > 0:
		# read only specified columns
		readCols = [dateTimeCol]
		if len(regionCol) > 0:
			readCols += [regionCol]
		readCols += serCols
		df = pd.read_csv(dataFile, parse_dates=[dateTimeCol], infer_datetime_format=True, usecols=readCols, na_values=missingStr)
	else:
		# read all columns
		df = pd.read_csv(dataFile, parse_dates=[dateTimeCol], infer_datetime_format=True, na_values=missingStr)

	print("Processing the input data.")

	if len(parInput['series_rename']) > 0:
		# NB: does not check whether the columns actually exist
		df.rename(columns=parInput['series_rename'], inplace=True)
	
	if dateTimeCol != 'DateTime':
		# the rest of the code requires that the column's name is DateTime
		df.rename(columns={dateTimeCol: 'DateTime'}, inplace=True)
		dateTimeCol = 'DateTime'

	if len(regionCol) > 0:
		# create columns for all combinations of region and series
		df = df.pivot(index=dateTimeCol, columns=regionCol)
		# this creates columns with names as tuples, which would create problems later
		# -> convert to <region>_<series>
		df.columns = [n[0] + '_' + n[1] if isinstance(n, tuple) else n for n in df.columns]
		# drop completely empty columns (in case not all combinations make sense)
		df.dropna(axis='columns', how='all', inplace=True)
		# also drop deterministic columns, since they have undefined correlations etc
		#  - this removes only series that are completely deterministic
		#  - later, we do the same for the actual series used for generation, for each season
		detCols = [c for c in df.columns if df[c].std() == 0]
		if len(detCols) > 0:
			print(" - removing {} deterministic column{}:".format(len(detCols), 's' if len(detCols) > 1 else ''))
			for c in detCols:
				print("   : {} = {}".format(c, df[c].mean()))
			df.drop(columns=detCols, inplace=True)

	# filter out dates with incomplete data
	df['Date'] = df.index.date
	if scenLen == 'day':
		hCount = df.groupby('Date').count().min(axis='columns')
		if float(pd.__version__[2:]) >= 24:
			df = df.merge(hCount.rename('hcount'), on='Date', right_index=True)  # .rename needed, as hCount does not have a name
		else:
			df = df.merge(hCount.rename('hcount').to_frame(), on='Date', right_index=True)  # older pandas cannot merge DataFrame with Series
		df = df[df['hcount'] == 24]  # TMP: this assumes hourly steps!
	elif scenLen == 'week':
		# each week can be uniquely identified using 'year_week-number'
		df['y_w'] = df.apply(lambda row: f'{row.name.year}_{row.name.week}', axis=1)
		hCount = df.groupby('y_w').count().min(axis='columns')
		if float(pd.__version__[2:]) >= 24:
			df = df.merge(hCount.rename('hcount'), on='y_w', right_index=True)  # .rename needed, as hCount does not have a name
		else:
			df = df.merge(hCount.rename('hcount').to_frame(), on='y_w', right_index=True)  # older pandas cannot merge DataFrame with Series
		df = df[df['hcount'] == 168]  # TMP: this assumes hourly steps!
		del df['y_w']
	else:
		assert False, "should not happen in a valid JSON file"
	del df['hcount']

	return df


def eval_scenarios(season: str, scens: pd.DataFrame, data: pd.DataFrame, prob=None, output=sys.stdout, errDict=0) -> bool:
	"""
	compute and print moments and correlations of scenarios
	and compare them to the target distribution 
	returns True on success, false otherwise
	"""
	
	with redirect_stdout(output):
		print("---------------------------\n{}\n---------------------------".format(str(season).upper()))
	
	try:
		scStat = sgc.df_moments(scens, prob) # scenario moments
	except Exception as e:
		print(f"ERROR: calculation of moments failed: {e}")
		return False
	tgStat = sgc.df_moments(data, None)  # target moments (equiprobable)
	difM = abs(scStat - tgStat)
	difM = difM.append(difM.max().rename('Max. diff.'))
	with redirect_stdout(output):
		# print arbitrarily large dataframe: https://stackoverflow.com/a/30691921/842693
		with pd.option_context('display.max_rows', None, 'display.max_columns', None):
			print("\nMOMENTS")
			print("\nTarget moments:\n", tgStat)
			print("\nScenario moments:\n", scStat)
			print("\nAbs. difference:\n", difM)

	# correlations undefined if we have constant columns -> remove them
	detCols = [c for c in scens.columns if scStat.loc[c]['std'] == 0]
	scCorr = scens.drop(columns=detCols, inplace=False).corr()
	tgCorr = data.drop(columns=detCols, inplace=False).corr()
	difC = abs(scCorr - tgCorr)
	difC = difC.append(difC.max().rename('Max. diff.'))
	with redirect_stdout(output):
		with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1024):
			print("\nCORRELATIONS")
			if len(detCols) > 0:
				print(f"- removed {len(detCols)} deterministic column{'s' if len(detCols) > 1 else ''} from data!")
			print("\nTarget correlations:\n", tgCorr)
			print("\nScenario correlations:\n", scCorr)
			print("\nAbs. difference:\n", difC)
	
	if isinstance(errDict, dict):
		# save errors for plotting later
		# - since there are more values for correlations, using a list of arrays
		diff = []
		difM = scStat - tgStat
		for c in difM.columns:
			diff.append(difM[c].values)
		C = (scCorr - tgCorr).reset_index(drop=True).values
		diff.append(np.array([C[i,j] for i in range(C.shape[0]) for j in range(i)]))
		errDict[season] = diff

	return True  # success


def process_data(data: pd.DataFrame, serLenH: int, aggrH: int, minSerStd = 0.0) -> pd.DataFrame:
	"""
	create a new dataframe from the data, taking into account:
	- serLenH = the length of the series (supported values are 24 and 168)
	- aggrH = period aggregation (number of hours in each evaluated period)
	- minSerStd = series with standard deviation less than this are considered deterministic

	OBS: no check for missing values!
	"""

	# OBS: since df is a basically a reference, the commands bellow will update the sender,
	#      BUT this stops at `data = data[...]`, which creates a copy
	#    : as a result, the sender would end-up halv-updated!
	#    : -> make a copy right at the start, and then return the result!
	df = data.copy()
	
	series = df.columns.tolist()
	if serLenH == 24:
		df['start'] = df.apply(lambda row: 1 if row.name.hour == 0 else 0, axis=1)
		df['seqHour'] = df.index.hour
	elif serLenH == 168:
		# note: .dayofweek is indexed from zero, starting on Monday
		df['start'] = df.apply(lambda row: 1 if row.name.hour == 0 and row.name.dayofweek == 0 else 0, axis=1)
		df['seqHour'] = 24 * df.index.dayofweek + df.index.hour
	else:
		assert False, "unsupported value of serLenH"
	df['seq'] = df['start'].cumsum()
	del df['start']

	# remove incomplete sequences at start and end
	# - there should be no holes in the middle as these are removed at the end of get_data()
	if sum(df['seq'] == 0) < serLenH:  # sum converts boolean to 0/1
		# the first sequence is incomplete -> delete
		df = df[df['seq'] > 0]
	if sum(df['seq'] == max(df['seq'])) < serLenH:  # sum converts boolean to 0/1
		# the last sequence is incomplete -> delete
		df = df[df['seq'] < max(df['seq'])]
	
	# aggregate time periods
	df['per'] = df['seqHour'] // aggrH
	pers = set(df['per'])

	aggFunc = {s: 'mean' for s in series}  # use average values for all series
	aggFunc['Date'] = 'min'  # use the first date -> will give the start date
	df = df.groupby(['seq', 'per']).agg(aggFunc)
	# now, we have a mult-index (seq, per) -> transform 'per' into columns:
	df = df.unstack()  # results in a multi-index on columns
	# delete reduntant date columns
	for per in pers - {0}:
		del df['Date', per]
	# flatten the rest - cannot use join(), since col[0] is a string a col[1] is a number
	df.columns = [f'{col[0]}_{col[1]}' for col in df.columns]
	# rename Date_0 to Date and set is as an index
	df.rename(columns={'Date_0': 'Date'}, inplace=True)
	df.set_index('Date', inplace=True)

	# remove all deteministic columns from the data
	# - this depends on season and aggregation: for ex., solar power in winter
	#   will have some constant (zero) series during the night
	# - NB: df.std() computes std() for numerical columns only, as a pandas Series
	detCols = [c for c, s in df.std().items() if s == 0]
	if len(detCols) > 0:
		print(f" - ignoring {len(detCols)} deterministic column{'s' if len(detCols) > 1 else ''}")
		# printing only in debug mode, since the list can be long:
		# : solar: 9 series * 10 hours per night * 7 days in a week = 630 zero series
		if logger.isEnabledFor(logging.DEBUG):
			maxDetColNameLen = max(len(c) for c in detCols)
			for c in detCols:
				print(f"   : {c:{maxDetColNameLen}s} = {df[c].mean():8.6f}")
		df.drop(columns=detCols, inplace=True)
	if minSerStd > 0:
		# in addition, remove 'almost deterministic' columns
		# - done separately, since we want to print these
		detCols = [c for c, s in df.std().items() if s < minSerStd]
		if len(detCols) > 0:
			print(f" - ignoring {len(detCols)} `almost deterministic' column{'s' if len(detCols) > 1 else ''}")
			maxDetColNameLen = max(len(c) for c in detCols)
			for c in detCols:
				print(f"   : {c:{maxDetColNameLen}s} = {df[c].mean():8.6f} ± {df[c].std():8.6f}")
			df.drop(columns=detCols, inplace=True)
	dim = len(df.columns)  # 'Date' is an index now -> all columns are data
	print(f" - scenario generation will use {dim} column{'s' if dim > 1 else ''}")

	return df


def main():

	# COMMAND-LINE ARGUMENTS
	# adding a custom help formatter, to allow for avoid line splits: https://stackoverflow.com/a/52606755/842693
	termWidth = shutil.get_terminal_size()[0]
	helpFormatter = lambda prog: argparse.HelpFormatter(prog, max_help_position=40, width=termWidth)
	parser = argparse.ArgumentParser(description='Scenario generator for EMPIRE and TIMES models', formatter_class=helpFormatter)
	# arguments
	parser.add_argument("-c", "--config", default="scen-gen.json", help="name of the configuration JSON file")
	parser.add_argument("-s", "--nmb-scen", type=int, help="number of scenarios to generate")
	parser.add_argument("-m", "--method", choices=['optimization', 'k-means', 'sampling', 'Wasserstein'], metavar="SELECTOR", help="selector method")
	parser.add_argument("-a", "--aggreg", type=int, help="hour aggregation")
	parser.add_argument("-o", "--output", help="output file name, without extension")
	parser.add_argument("-T", "--max-time", type=int, help="time limit for the optimization method [s]")
	parser.add_argument("--n-samples", type=int, help="number of samples for the sampling method")
	parser.add_argument("--k-means-var", help="variant of the k-means method to use")
	# parsing
	args = parser.parse_args()

	configFile = args.config

	params = read_config(configFile)
	# overwrite config-file data with command-line parameters (if given)
	if args.nmb_scen is not None:
		params['scen-gen']['nmb-scen'] = args.nmb_scen
	if args.method is not None:
		params['scen-gen']['selector'] = args.method
	if args.aggreg is not None:
		params['scen-gen']['aggreg-hours'] = args.aggreg
	if args.output is not None:
		params['output']['filename-base'] = args.output
	if args.max_time is not None:
		params['scen-gen']['optimization']['max-time'] = args.max_time
	if args.k_means_var is not None:
		params['scen-gen']['k-means']['variant'] = args.k_means_var
	if args.n_samples is not None:
		params['scen-gen']['sampling']['nmb-samples'] = args.n_samples
		params['scen-gen']['sampling-WD']['nmb-samples'] = args.n_samples

	# read the input data
	df = get_data(params)

	# process the rest of the configuration file
	parSG = params['scen-gen'] # type: dict
	nScen = parSG['nmb-scen']
	scenLenS = parSG['scen-length']  # string (day/week)
	ScenLenH = {
		'day': 24,
		'week': 168
	}
	scenLenH = ScenLenH[scenLenS]  # length in hours; NB: assumes hourly series!
	hAggreg = parSG.get('aggreg-hours', 1)
	minSerStd = parSG.get('min-series-std', 0)

	selectorType = parSG.get('selector', 'optimization')
	print(f"Initializing selector object of type '{selectorType}'.")
	if selectorType == 'optimization':
		from scen_select_optimize import SelectByOptimize
		selector = SelectByOptimize(parSG)
	elif selectorType == 'k-means':
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
		from scen_select_sampling_WD import SelectBySamplingWD
		selector = SelectBySamplingWD(parSG)
	else:
		assert False, f"unsupported selector type '{selectorType}'"

	parOut = params['output'] # type: dict
	outFileBase = parOut['filename-base']
	makeStatFile = parOut.get('store-scen-stat', False)
	outStatFile = outFileBase + '_stat.out' if makeStatFile else ""
	errPlotPng = parOut.get('error-plot_png', False)
	errPlotPdf = parOut.get('error-plot_pdf', False)
	makeErrorPlot = errPlotPng or errPlotPdf
	saveRawScen = parOut.get('save-raw-output', False)

	# add a column for season
	# - using meteorological seasons (https://en.wikipedia.org/wiki/Season#Meteorological)
	# - winter is December to February, etc.
	Seasons = ['winter', 'spring', 'summer', 'autumn']
	df['Season'] = df['Date'].apply(lambda d : Seasons[(d.month // 3) % 4])
	if scenLenS == 'week':
		# align seasons by weeks - each week gets the season of its Thursday
		# OBS: double [[]] below ensures that sdf is a DataFrame
		sdf = df[['Season']][df.apply(lambda row: row.name.dayofweek==3 and row.name.hour==0, axis=1)]
		sdf['yw'] = sdf.apply(lambda row: f'{row.name.year}_{row.name.week}', axis=1)
		sDict = sdf.set_index('yw').to_dict()['Season']
		# OBS: a week can be missing from sDict, for ex. around the new year
		df['Season'] = df.apply(lambda row: sDict.get(f'{row.name.year}_{row.name.week}', row['Season']), axis=1)
		del sdf
	seasons = df.groupby('Season').nunique().index.tolist()  # in case data is missing some...

	# file for scenario statistics
	if len(outStatFile) > 0:
		outStatF = open(outStatFile, 'w')
	else:
		outStatF = open(os.devnull, 'w')

	# per-season output
	dSRes = dict()
	dErrPlt = dict()

	for s in seasons:
		print(f"Generating scenarios for season `{s}':")
		dfs = df[df['Season'] == s]
		del dfs['Season']
		
		# compute statistics
		print(" - processing the input data")
		dAvg = process_data(dfs, scenLenH, hAggreg, minSerStd)
		# with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 200):
		# 	print("\nTMP: dAvg =\n", dAvg, '\n')
		print(f" - found {len(dAvg.index)} relevant {scenLenS}s in data")

		# calling the run() method of the selector
		tStart = timer()
		[resProb, resStatus] = selector.run(dAvg, season=s)
		if len(resProb) == 0:
			# no results!
			print(f" - selection failed, status = {resStatus}")
			print(" -> ABORTING generation for this season!\n")
			continue
		print(f" - selection finished in {timer() - tStart:.1f} s")
		print(f" - returned status: {resStatus}")

		if len(resProb) != nScen:
			print(" - ERROR: wrong number of scenarios returned -> aborting!")
			print(f"nScen = {nScen}")
			print(f"len(resProb) = {len(resProb)}")
			print(f"resProb = {resProb}")
			continue

		# filter dAvg by index: https://stackoverflow.com/a/45040370/842693
		# - note: in case of multiindex: https://stackoverflow.com/a/25225009/842693
		# - filtering creates a read-only slice -> need to create a copy!
		dSel = dAvg[dAvg.index.isin(resProb.keys())].copy()
		if len(dSel) != nScen:
			print(" - ERROR: wrong number of non-zero probabilities returned -> aborting!")
			continue

		# compute and output statistics
		if len(outStatFile) > 0 or makeErrorPlot:
			print(" - computing and saving statistics")
			prob = list(resProb.values())
			if makeErrorPlot:
				# fills dErrPlt with extra data for plotting
				evalOK = eval_scenarios(s, dSel, dAvg, prob, outStatF, dErrPlt)
				if not evalOK:
					print("WARNING: scenario evaluation failed -> cannot generate output plots")
					makeErrorPlot = False
			else:
				eval_scenarios(s, dSel, dAvg, prob, outStatF)
			if len(outStatFile) > 0:
				print(file=outStatF)
				outStatF.flush()

		df_sc = [None] * nScen
		for sc, sc_t0 in enumerate(dSel.index):
			# get rows from dfs corresponding to this scenario
			t0 = np.datetime64(sc_t0)  # needed to comparing with dfs.index
			df_sc[sc] = dfs[(dfs.index >= t0) & (dfs.index < t0 + np.timedelta64(scenLenH, 'h'))].reset_index()
			df_sc[sc]['scen'] = sc + 1  # scenario number, starting from 1
		# concatenate/merge the data frames; keep the original index as 'hour'
		dRes = pd.concat(df_sc).reset_index().rename(columns={'index': 'hour'})
		del dRes['Date']    # no longer needed
		dRes['Season'] = s  # required later
		
		if saveRawScen:
			outFile = f"{outFileBase}_{s}.csv"
			print(f" - saving results to file {outFile}")
			dRes.to_csv(outFile, float_format="%g", date_format="%Y-%m-%d %H:%M", index=False)

		# store for later
		dSRes[s] = dRes

	print("Merging and processing the results.")

	# concatenate/merge all seasons into one dataframe
	dRes = pd.concat([dSRes[s] for s in seasons if s in dSRes]).reset_index(drop=True)

	# convert to values/formats required by TIMES:
	sCode = {'winter': 'WI', 'spring': 'SP', 'summer': 'SU', 'autumn': 'FA'}
	dRes['Scenario'] = ['SW2-{:03}'.format(sc) for sc in dRes['scen']]
	dRes['TimeSlice'] = dRes.apply(lambda row: f"{sCode[row['Season']]}_{row['hour']+1:02}", axis = 1)
	dRes.drop(columns=['Season', 'scen', 'DateTime', 'hour'], inplace=True)
	dRes = pd.melt(dRes, id_vars=['TimeSlice','Scenario'], var_name='tech_reg', value_name='Production')
	dRes['Region'] = dRes['tech_reg'].apply(lambda tr: tr.split('_')[1])
	dRes['Technology'] = dRes['tech_reg'].apply(lambda tr: tr.split('_')[0])
	del dRes['tech_reg']

	# get the total production and normalize the scenario values
	dResTot = dRes.groupby(['Scenario','Technology','Region']).sum()
	dResAvg = dRes.groupby(['Scenario','Technology','Region']).mean()  # must be done here, before we scale dRes!
	dRes = dRes.join(dResTot, on=['Scenario','Technology','Region'], rsuffix='Tot') # alt: dRes = dRes.merge(dResTot, on=['Scenario','Technology','Region'], suffixes=('', 'Tot'))
	dRes.Production /= dRes.ProductionTot
	# to check that the aggregation worked: dRes.groupby(['Scenario','Technology','Region']).sum()['Production'].describe()
	del dRes['ProductionTot']
	del dResTot

	# add param-type and concatenate the two tables
	dRes['Param'] = 'S_COM_FR'
	dResAvg['Param'] = 'S_NCAP_AFS'
	dRes = dResAvg.reset_index().append(dRes, ignore_index=True, sort=False) # alt: dRes = pd.concat([dRes, dResAvg.reset_index()], ignore_index=True, sort=False)
	del dResAvg

	dRes = dRes[['Param','Region','Technology','TimeSlice','Scenario','Production']]
	outFile = outFileBase + ".csv"
	print("Saving results to file " + outFile + '.')
	dRes.to_csv(outFile, float_format="%g", index=False)

	if makeErrorPlot:
		print("Creating the error figure.")
		import matplotlib.pyplot as plt

		seasonList = ['spring', 'summer', 'autumn', 'winter']  # (seasons is sorted alphabetically)
		pltSeasons = [s for s in seasonList if s in dErrPlt]
		serTypes = ['mean', 'std', 'skew', 'kurt', 'corr']
		data = []       # all series in one array
		serSeason = []  # season of each series
		serType = []    # type of each series
		for s in pltSeasons:
			st = 0
			for ser in dErrPlt[s]:
				if len(ser) > 0:
					data.append(ser)
					serSeason.append(s)
					serType.append(serTypes[st])
					st += 1

		nSer = len(data)
		cols = range(1, nSer + 1)

		# initialize the matplotlib figure
		fig = plt.figure(figsize=(8,4))  # default size is 6.4x4.8 (inches)
		ax = fig.gca()  # gca = get current axes; creates one if necessary
		# alt: fig, ax = plt.subplots()

		# format axes
		ax.set_xlim(0.25, nSer + 0.75)
		ax.set_xticks(cols) 
		ax.set_xticklabels(serType)
		ax.tick_params(axis='x', rotation=90)
		ax.axhline(color='grey',lw=0.25,alpha=0.5)  # thin line at horizontal axis (y=0)
		#
		ax.set_ylim(-1.1, 1.1)

		# create and format the violin plots
		## TMP: there is a bug(?) in violinplot(), which emits the following VisibleDeprecationWarning:
		##      "Creating an ndarray from ragged nested sequences
		##       (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes)
		##       is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray
		##    : the following two lines disable the warning
		import warnings
		warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
		#
		vp = ax.violinplot(data, cols, widths=0.8, showmeans=True, showextrema=True, showmedians=True)
		#
		seasonColor = {'spring': 'green', 'summer': 'red', 'autumn': 'orange', 'winter': 'blue'}
		for p in range(nSer):
			vp['bodies'][p].set_color(seasonColor[serSeason[p]])
		#
		for key in ['cmeans', 'cmins', 'cmaxes', 'cbars', 'cmedians']:
			vp[key].set_color('xkcd:charcoal')
			vp[key].set_linewidth(0.5)
			vp[key].set_alpha(0.9)
		#
		vp['cmeans'].set_lw(1.0)

		# save the results
		if errPlotPng:
			fig.savefig(outFileBase + '.png', bbox_inches='tight', dpi=300)
		if errPlotPdf:
			fig.savefig(outFileBase + '.pdf', bbox_inches='tight')

	print("Done.")


if __name__ == "__main__":
	""" This is executed when run from the command line """
	main()
