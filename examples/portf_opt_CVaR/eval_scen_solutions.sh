#!/bin/bash

# solve all the 'base cases' with full data sets

SOLVER=highs
#SOLVER_PATH=

DATA="prices/prices_*.csv"
SOL_DIR=scens
FORMAT="ColSep=tab IndexCol=date"
RISK_W=(0 0.1 0.2 0.3 0.4 0.5)
RES_FILE="$(basename "$0" .sh).csv"

if [ -f "$RES_FILE" ]; then
	echo
	echo "Found existing $RES_FILE - appending!"
else
	# create header
	echo "solution-file	is_obj-func	is_exp-val	is_cvar	oos_obj-func	oos_exp-val	oos_cvar" > "$RES_FILE"
fi

echo "Start: $(date)"
echo

for d in $DATA; do
	dataFile=$(basename "$d" .csv)
	solBase="$SOL_DIR/${dataFile/prices/case}"
	for r in "${RISK_W[@]}"; do
		for s in "${solBase}_"*"_w${r}.sol"; do
			if [ -f "$s" ]; then
				echo
				# Check if solution is already in results file
				if grep -q "$(basename "$s" .sol)" "$RES_FILE"; then
					echo "Evaluation of $s found in $RES_FILE - skipping"
				else
					echo "Evaluating solution from $s on data file $d"
					mosel exec -g portfolio_opt.mos RiskW="$r" ScenFile="$d" $FORMAT EvalSolFile="$s" EvalResFile="$RES_FILE" Solver="$SOLVER" SolverPath="$SOLVER_PATH"
				fi
			fi
		done
	done
done

echo
echo "Finish: $(date)"
