#!/bin/bash

# solve all the 'base cases' with full data sets

SOLVER=highs
SOLVER_PATH=

DATA="prices/prices_*.csv"
FORMAT="ColSep=tab IndexCol=date"
RISK_W=(0 0.1 0.2 0.3 0.4 0.5)
RES_DIR=prices


echo "Start: $(date)"
echo

for d in $DATA; do
	inFile=$(basename "$d" .csv)
	outBase="${inFile/prices/case}"
	for r in "${RISK_W[@]}"; do
		mosel exec -g portfolio_opt.mos RiskW="$r" ScenFile="$d" $FORMAT SolFile="$RES_DIR/${outBase}_opt_w${r}.sol" Solver="$SOLVER" SolverPath="$SOLVER_PATH"
	done
done

echo
echo "Finish: $(date)"
