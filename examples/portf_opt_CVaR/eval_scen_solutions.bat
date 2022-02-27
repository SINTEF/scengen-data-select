@echo off

:: solve all the 'base cases' with full data sets

setlocal enabledelayedexpansion

set SOLVER=cbc
set SOLVER_PATH=cbc.bat

set DATA="prices\prices_*.csv"
set SOL_DIR=scens
set FORMAT=ColSep=tab IndexCol=date
set RISK_W=(0, 0.1, 0.2, 0.3, 0.4, 0.5)
set RES_FILE=%~n0.csv

if exist %RES_FILE% (
	echo:
	echo Found existing %RES_FILE% - appending!
) else (
	:: create header
	echo solution-file	is_obj-func	is_exp-val	is_cvar	oos_obj-func	oos_exp-val	oos_cvar>%RES_FILE%
)

echo Start: %date% %time%
echo:

for %%d in (%DATA%) do (
	set dataFile=%%~nd
	set solBase=%SOL_DIR%\!dataFile:prices=case!
	for %%r in %RISK_W% do (
		for %%s in (!solBase!_*_w%%r.sol) do (
			if exist %%s (
				echo:
				REM see https://stackoverflow.com/a/13476936/842693
				>nul find "%%~ns" %RES_FILE% && (
					echo Evaluation of %%s found in %RES_FILE% - skipping
				) || (
					echo Evaluating solution from %%s on data file %%d
					mosel exec -g portfolio_opt.mos RiskW=%%r ScenFile=%%d %FORMAT% EvalSolFile=%%s EvalResFile=%RES_FILE% Solver=%SOLVER% SolverPath=%SOLVER_PATH%
				)
			)
		)
	)
)

echo:
echo Finish: %date% %time%

endlocal
