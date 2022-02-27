import os
import glob
import pandas as pd
import shutil
import argparse
#import pickle
import matplotlib.pyplot as plt
import numpy as np

# COMMAND-LINE ARGUMENTS
# adding a custom help formatter, to allow for avoid line splits: https://stackoverflow.com/a/52606755/842693
termWidth = shutil.get_terminal_size()[0]
helpFormatter = lambda prog: argparse.HelpFormatter(prog, max_help_position=40, width=termWidth)
parser = argparse.ArgumentParser(description='Comparing errors in scenarios for the TIMES model', formatter_class=helpFormatter)
# arguments
parser.add_argument("scen_dir", type=str, nargs="?", default=".", help="folder with scenario-generation log file(s)")
parser.add_argument("-m", "--mask", type=str, default="gen_scen*.log", help="file mask for the log file(s)")
parser.add_argument('-f', '--format', type=str, default='png', choices=['png','pdf'], help='output format')
parser.add_argument('-s', '--scale', type=float, default=0.6, help='default scaling of the figures')
parser.add_argument('-o', '--out-dir', type=str, help='output folder, if different from scen_dir')
parser.add_argument("--tex", action='store_true', help="use TeX to render text - slow, needs TeX")
# parsing
args = parser.parse_args()

fig_ext = '.' + args.format
fig_scale = args.scale
render_text_by_TeX = args.tex
out_dir = args.scen_dir if args.out_dir is None else args.out_dir

files = glob.glob(args.scen_dir + '/' + args.mask)

# collect all data to a dictionary, then convert it to DataFrame
# - should be faster than adding one row at a time
# - moreover, it preserves integers, while row-wise building creates floats
dfDict = dict()
dfIdx = 0
for file in files:
	with open(file, 'r') as f:
		print(f'\n{file}:')
		lines = f.readlines()
		nLines = len(lines)
		l = 0
		while (l < nLines):
			while l < nLines and "selection finished in" not in lines[l]:
				l += 1
			if l < nLines:
				assert "selection finished in" in lines[l], "logic check"
				genTime = float(lines[l].split()[-2])
				l += 2  # should give line with 'Saving results to file ...'
				if 'Saving results to file' in lines[l]:
					scenFile = lines[l].split()[-1][:-1]
					
					# remove path and extension
					scenName = os.path.splitext(os.path.basename(scenFile))[0]
					scen = scenName.split('_')
					if len(scen) == 7:
						dfRow = {
							'perLen': scen[1],
							'perType': scen[2],
							'nVar': int(scen[3]),
							'method': scen[4],
							'nScen': int(scen[5][:-1]),
							'tree': int(scen[6]),
							'genTime': genTime
						}
						dfDict[dfIdx] = dfRow
						dfIdx += 1
					else:
						print(f"Warning: line[{l}] does not include valid scen. file: '{lines[l][:-1]}'")
					l += 1
				else:
					print(f"Warning: line[{l}] does not include 'Saving results to...', as expected")

dfs = pd.DataFrame.from_dict(dfDict, orient='index')
#print(dfs)
dfs['per'] = dfs.perLen + '_' + dfs.perType

Pers = set(dfs.per.values)
NVars = set(dfs.nVar.values)
Methods = set(dfs.method.values)

if render_text_by_TeX:
	# can also specify font family and font, for ex.:
	# "font.family": "sans-serif", "font.sans-serif": ["Helvetica"]
	plt.rcParams.update({
		"text.usetex": True
	})

for p in Pers:
	dfp = dfs[dfs.per == p]
	for m in Methods:
		dfm = dfp[dfp.method == m]


		# plot with nVar as series and nScen on x-axis
		df = pd.pivot_table(dfm, columns='nScen', values='genTime', aggfunc=np.average, index='nVar')
		ax = df.plot(style='-o')
		ax.set_xlabel('dimension of the data vector')
		ax.legend(title='scenarios')
		ax.xaxis.set_ticks(range(10,25+1,5))
		fig = plt.gcf()
		figSize = fig.get_size_inches() * fig_scale
		fig.set_size_inches(figSize)
		plt.tight_layout()
		plt.savefig(f'{out_dir}/gen-time_{p}_{m}_per-nscen' + fig_ext, bbox_inches='tight', pad_inches=0.02)

		# plot with nScen as series and nVar on x-axis
		df = pd.pivot_table(dfm, columns='nVar', values='genTime', aggfunc=np.average, index='nScen') 
		ax = df.plot(style='-o')
		ax.set_xlabel('number of scenarios')
		ax.legend(title='dimension')
		plt.gcf().set_size_inches(figSize)
		plt.tight_layout()
		plt.savefig(f'{out_dir}/gen-time_{p}_{m}_per-nvar' + fig_ext, bbox_inches='tight', pad_inches=0.02)
