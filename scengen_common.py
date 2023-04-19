# definitions common for all scenario-selection methods

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


def exp_powers(df: pd.DataFrame, powers: list) -> pd.DataFrame:
	"""
	for a given dataframe, compute expected powers of all series
	and return them as a dataframe with the powers in the index
	"""
	dEPwr = pd.DataFrame()
	for p in powers:
		dEPwr[p] = (df**p).mean()
	return dEPwr

def ser_moments(ser: pd.Series, prob = None) -> list:
	"""
	get the first four moments of pandas series, 
	- unspecified prob implies equiprobable values
	"""
	mean = np.average(ser, weights=prob)
	ser0 = ser - mean
	std = np.sqrt(np.average(ser0**2, weights=prob))
	if std == 0:
		return [mean, std, 0, 0]
	skew = np.average(ser0**3, weights=prob) / std**3
	kurt = np.average(ser0**4, weights=prob) / std**4
	return [mean, std, skew, kurt]

def df_moments(df: pd.DataFrame, prob = None) -> pd.DataFrame:
	"""
	create a dataframe with the first four moments of a given dataframe
	- unspecified prob implies equiprobable values
	"""
	mdict = dict()
	for c in df.columns:
		mdict[c] = ser_moments(df[c], prob)
	mdf = pd.DataFrame.from_dict(mdict, orient='index', columns=['mean', 'std', 'skew', 'kurt'])
	return mdf


class SelectorBase(ABC):
	"""
	abstract base class (ABC) for all the Selector classes
	"""

	def __init__(self, parSG: dict):
		"""
		arguments:
		- parSG - contents of the 'scen-gen' part of the JSON file, as dict
		"""
		self._logFileBase = parSG.get('logfile-base', '')  # can be overwritten
		self._nmbScen = parSG['nmb-scen']
	
	@abstractmethod
	def run(self, df: pd.DataFrame, season = '', nScen: int = None) -> [dict, str]:
		"""
		this runs the selection method
		- abstract method, so all derived methods have to implement it

		arguments:
		- df = data frame with the data series; its index is used to identify the selection
		- season = name of the season - used for output
		- nScen = number of scenarios/sequences to select (if different from self._nmbScen)

		returns:
		- dictionary index->probability for the selected value
			- must include exactly nScen items
			- empty dictionary is used to denote failure
		- return status, as a string
		"""
		pass
	
