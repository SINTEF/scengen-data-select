#!/bin/bash

# solve all the 'base cases' with full data sets

SOLVER=highs
#SOLVER_PATH=

SCEN_DIR=scens
SCEN_FILES="scen_*.csv"
FORMAT="ColSep=,"
RISK_W=(0 0.1 0.2 0.3 0.4 0.5)
PROB_COL_STR="ProbCol=prob"

echo "Start: $(date)"
echo

for s in $SCEN_DIR/$SCEN_FILES; do
	inFile=$(basename "$s" .csv)
	outBase="$SCEN_DIR/${inFile/scen_/case_}"
	for r in "${RISK_W[@]}"; do
		if [ ! -f "${outBase}_w${r}.sol" ]; then
			echo
			echo "Solving $s with risk-weight $r"
			echo
			mosel exec -g portfolio_opt.mos RiskW="$r" ScenFile="$s" $FORMAT SolFile="${outBase}_w${r}.sol" $PROB_COL_STR Solver="$SOLVER" SolverPath="$SOLVER_PATH"
		fi
	done
done

echo
echo "Finish: $(date)"
