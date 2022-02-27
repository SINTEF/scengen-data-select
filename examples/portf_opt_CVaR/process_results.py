import glob
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.collections import PolyCollection
import numpy as np
import seaborn as sns
import shutil
import argparse
import os

# COMMAND-LINE ARGUMENTS
# adding a custom help formatter, to allow for avoid line splits: https://stackoverflow.com/a/52606755/842693
termWidth = shutil.get_terminal_size()[0]
helpFormatter = lambda prog: argparse.HelpFormatter(prog, max_help_position=40, width=termWidth)
parser = argparse.ArgumentParser(description='Comparing errors in scenarios for the TIMES model', formatter_class=helpFormatter)
# arguments
parser.add_argument("eval_res_dir", type=str, nargs="?", default=".", help="folder with file 'eval_scen_solutions.csv'")
parser.add_argument("-r", "--ref-sol-dir", type=str, default="prices", help="folder with reference solutions")
parser.add_argument("-m", "--mask", type=str, default="case_*_opt_*.sol", help="file mask for ref. solution files")
parser.add_argument('-i', '--input-file', type=str, default='eval_scen_solutions.csv', help='input file with evaluated scen. files')
parser.add_argument('-f', '--format', type=str, default='png', choices=['png','pdf'], help='output format')
parser.add_argument('-s', '--scale', type=float, default=0.8, help='default scaling of the figures')
parser.add_argument('-o', '--out-dir', type=str, help='output folder, if different from eval_res_dir')
parser.add_argument("--tex", action='store_true', help="use TeX to render text - slow, needs TeX")
# parsing
args = parser.parse_args()

fig_ext = '.' + args.format
fig_scale = args.scale
render_text_by_TeX = args.tex
out_dir = args.eval_res_dir if args.out_dir is None else args.out_dir
res_file = args.eval_res_dir + '/' + args.input_file

optSolFiles = glob.glob(args.ref_sol_dir + '/' + args.mask)
sdf = pd.DataFrame()
for file in optSolFiles:
	dfRow = {'sol-file': os.path.basename(file)}
	with open(file, 'r') as f:
		for l in f.readlines():
			if len(l.split()) == 2:
				kw = l.split()[0]
				if kw in {'obj_f', 'exp_v', 'cvar'}:
					dfRow[kw] = float(l.split()[1])
	sdf = sdf.append(dfRow, ignore_index=True)

# add columns based on 'sol-file'
# example value: case_1w_all_10_opt_w0.5.sol
sdf['dim'] = sdf['sol-file'].apply(lambda f: int(f.split('_')[3]))
# NB: the last part includes the '.sol' as well!
sdf['risk-weight'] = sdf['sol-file'].apply(lambda f: float(f.split('_')[-1][1:-4]))


df = pd.read_csv(res_file, delimiter='\t')

def method_label(m: str) -> str:
	"""
	function to create method labels from their codes used in the output files

	OBS: if changed, need to apply plot_order as well!
	"""
	ms = m.split('-')
	if ms[0] == 'optimization':
		# assuming 'optimization-<T>', where T is the time in seconds
		return f"opt. {ms[1]} s"
	if ms[0] == 'sampling':
		if ms[1] == 'WD':
			# assuming 'sampling-WD-<N>', where N is the number of iterations
			return f"sampl. WD {ms[2]}x"
		elif ms[1] == '1':
			return "sampling 1x"
		else:
			# assuming 'sampling-<N>
			return f"sampl. MM {ms[1]}x"
	if ms[0] == 'Wasserstein':
		return "WD heuristic"
	if ms[0] == 'k':
		return "$k$-means"
	if m == 'scen-red':
		return "scen. reduction"
	return m

# add columns based on 'solution-file'
# example value: case_1w_all_10_Wasserstein_50s_7_w0.4.sol
df['dim'] = df['solution-file'].apply(lambda f: int(f.split('_')[3]))
df['sg-method'] = df['solution-file'].apply(lambda f: f.split('_')[4])
df['sg-method-label'] = df['sg-method'].apply(method_label)
df['nmb-scen'] = df['solution-file'].apply(lambda f: int(f.split('_')[5][:-1]))
df['iter'] = df['solution-file'].apply(lambda f: int(f.split('_')[6]))
# NB: the last part includes the '.sol' as well!
df['risk-weight'] = df['solution-file'].apply(lambda f: float(f.split('_')[-1][1:-4]))
#
df['opt-sol-file'] = df['solution-file'].apply(lambda f: '_'.join(os.path.basename(f).split('_')[0:4]) + '_opt_' + f.split('_')[-1])

sgMethods = set(df['sg-method'])

formatter = ScalarFormatter()
formatter.set_scientific(False)

for sgm in sgMethods:
	dfm = df[df['sg-method'] == sgm]
	ax = dfm.plot(
		x='oos_cvar', y='oos_exp-val', title=sgm,
		xlim=(-480, -40), ylim=(0.7, 13.3),
		xlabel='CVaR', ylabel='expected profit',
		kind='scatter', logx='sym', xticks=[-50,-100,-200,-300, -400],
		s=dfm['nmb-scen']/4 + 5, edgecolor='black', linewidth=0.5,
		c='risk-weight', colormap='viridis')
	ax.xaxis.set_major_formatter(formatter)

	# get all the relevant optimal solutions
	# solFiles = set(dfm['opt-sol-file'])
	# sdfm = sdf[sdf['sol-file'].isin(solFiles)]
	# sdfm.plot(ax=ax, x='cvar', y='exp_v',
	#           kind='scatter', marker='X', s=50, edgecolor='darkred',
	#           c='risk-weight', colormap='viridis', colorbar=False)

	ax.set_xlabel('CVaR')
	ax.set_ylabel('expected profit')
	#plt.show()
	plt.tight_layout()
	fig = plt.gcf()
	fig.set_size_inches(fig.get_size_inches() * fig_scale)
	plt.savefig(f'{out_dir}/res_eval_eff_{sgm}' + fig_ext, bbox_inches='tight', pad_inches=0.02)
	plt.close()

# the same, but separate for each dimension
# different limits for different dimensions!
ylims = {
	10: (0.8, 7.7),
	20: (1.5, 13.3),
	25: (0.7, 13.3)
}
for d in set(df['dim']):
	for sgm in sgMethods:
		sgml = method_label(sgm)
		# NB: one cannot use 'and' here!
		dfm = df[(df['dim'] == d) & (df['sg-method'] == sgm)]
		ax = dfm.plot(
			x='oos_cvar', y='oos_exp-val', title=f'dim = {d} - {sgml}',
			xlim=(-480, -40), ylim=ylims[d],
			xlabel='CVaR', ylabel='expected profit',
			kind='scatter', logx='sym', xticks=[-50,-100,-200,-300, -400],
			s=dfm['nmb-scen']/4 + 5, edgecolor='black', linewidth=0.5,
			c='risk-weight', colormap='viridis')
		ax.xaxis.set_major_formatter(formatter)

		# get all the relevant optimal solutions
		solFiles = set(dfm['opt-sol-file'])
		sdfm = sdf[sdf['sol-file'].isin(solFiles)]
		sdfm.plot(ax=ax, x='cvar', y='exp_v',
		          kind='scatter', marker='x', s=50, edgecolor='darkred',
		          c='risk-weight', colormap='viridis', colorbar=False)

		ax.set_xlabel('CVaR')
		ax.set_ylabel('expected profit')
		#plt.show()
		plt.tight_layout()
		fig = plt.gcf()
		fig.set_size_inches(fig.get_size_inches() * fig_scale)
		plt.savefig(f'{out_dir}/res_eval_eff_{d}_{sgm}' + fig_ext, bbox_inches='tight', pad_inches=0.02)
		plt.close()


# add the optimal results to df
df = df.merge(sdf, how='left', left_on='opt-sol-file', right_on='sol-file', suffixes=('', '_opt'))
df.drop(columns=['sol-file', 'dim_opt', 'risk-weight_opt'], inplace=True)
df.rename(columns={c : f'opt_{c}' for c in ['cvar', 'exp_v', 'obj_f']}, inplace=True)
df['oos_obj_err'] = df['opt_obj_f'] - df['oos_obj-func']

# pivot by method
#dfp = pd.pivot_table(df, columns='sg-method', values='oos_obj_err')
#dfp = df.pivot(index=['opt-sol-file', 'risk-weight', 'iter'], columns='sg-method', values='oos_obj_err')

# order of methods in the violin plots
# TODO: write a function
plot_order = [
	'WD heuristic',
	'opt. 300 s',
	'opt. 1800 s',
	'sampling 1x',
	'sampl. MM 500x',
	'sampl. WD 500x',
	'$k$-means',
	'scen. reduction'
]


sns.set_style("whitegrid")
ax = sns.violinplot(x=df['sg-method-label'], y=df['oos_obj_err'], cut=0, inner='box', order=plot_order, width=0.92, linewidth=1.3)  # default linewidth looks to be around 1.5
ax.set_xlabel('scenario-generation method')
ax.set_ylabel('')
ax.set_title('Discretization error')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
fig = plt.gcf()
fig.set_size_inches(fig.get_size_inches() * fig_scale)
plt.savefig(f'{out_dir}/res_eval_gap' + fig_ext, bbox_inches='tight', pad_inches=0.02)
ax.set_ylim(ymin=-0.5,ymax=8.5)
plt.tight_layout()
plt.savefig(f'{out_dir}/res_eval_gap_zoom' + fig_ext, bbox_inches='tight', pad_inches=0.02)
#plt.show()
plt.close()

for rw in set(df['risk-weight']):
	dfw = df[df['risk-weight'] == rw]
	ax = sns.violinplot(x=dfw['sg-method-label'], y=dfw['oos_obj_err'], cut=0, inner='box', order=plot_order)
	ax.set_xlabel('scenario-generation method')
	ax.set_ylabel('')
	ax.set_title(f'Discretization error with risk-weight = {rw}')
	plt.xticks(rotation=45, ha='right')
	plt.tight_layout()
	fig = plt.gcf()
	fig.set_size_inches(fig.get_size_inches() * fig_scale)
	plt.savefig(f'{out_dir}/res_eval_gap_w-{rw}' + fig_ext, bbox_inches='tight', pad_inches=0.02)
	#plt.show()
	plt.close()

for d in set(df['dim']):
	dfd = df[df['dim'] == d]
	ax = sns.violinplot(x=dfd['sg-method-label'], y=dfd['oos_obj_err'], cut=0, inner='box', order=plot_order)
	ax.set_xlabel('scenario-generation method')
	ax.set_ylabel('')
	ax.set_title(f'Discretization error with dim = {d}')
	plt.xticks(rotation=45, ha='right')
	plt.tight_layout()
	fig = plt.gcf()
	fig.set_size_inches(fig.get_size_inches() * fig_scale)
	plt.savefig(f'{out_dir}/res_eval_gap_dim-{d}' + fig_ext, bbox_inches='tight', pad_inches=0.02)
	#plt.show()
	plt.close()

for s in set(df['nmb-scen']):
	dfs = df[df['nmb-scen'] == s]
	ax = sns.violinplot(x=dfs['sg-method-label'], y=dfs['oos_obj_err'], cut=0, inner='box', order=plot_order)
	ax.set_xlabel('scenario-generation method')
	ax.set_ylabel('')
	ax.set_title(f'Discretization error with {s} scenarios')
	plt.xticks(rotation=45, ha='right')
	plt.tight_layout()
	fig = plt.gcf()
	fig.set_size_inches(fig.get_size_inches() * fig_scale)
	plt.savefig(f'{out_dir}/res_eval_gap_scen-{s}' + fig_ext, bbox_inches='tight', pad_inches=0.02)
	#plt.show()
	plt.close()
