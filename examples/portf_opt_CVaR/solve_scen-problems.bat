@echo off

:: solve all the 'base cases' with full data sets

setlocal enabledelayedexpansion

set SOLVER=highs
::set SOLVER_PATH=

set SCEN_DIR=scens
set SCEN_FILES="scen_*.csv"
set FORMAT=ColSep=,
set RISK_W=(0, 0.1, 0.2, 0.3, 0.4, 0.5)
set PROB_COL_STR=ProbCol=prob

echo Start: %date% %time%
echo:

for %%s in (%SCEN_DIR%\%SCEN_FILES%) do (
	set inFile=%%~ns
	set outBase=%SCEN_DIR%\!inFile:scen_=case_!
	for %%r in %RISK_W% do (
		if not exist !outBase!_w%%r.sol (
			echo:
			echo Solving %%s with risk-weight %%r
			echo:
			mosel exec -g portfolio_opt.mos RiskW=%%r ScenFile=%%s %FORMAT% SolFile=!outBase!_w%%r.sol %PROB_COL_STR% Solver=%SOLVER% SolverPath=%SOLVER_PATH%
		)
	)
)

echo:
echo Finish: %date% %time%

endlocal
