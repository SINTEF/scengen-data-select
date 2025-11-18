@echo off

:: solve all the 'base cases' with full data sets

setlocal enabledelayedexpansion

set SOLVER=highs
::set SOLVER_PATH=

set DATA="prices\prices_*.csv"
set FORMAT=ColSep=tab IndexCol=date
set RISK_W=(0, 0.1, 0.2, 0.3, 0.4, 0.5)
set RES_DIR=prices

echo Start: %date% %time%
echo:

for %%d in (%DATA%) do (
	set inFile=%%~nd
	set outBase=!inFile:prices=case!
	for %%r in %RISK_W% do (
		mosel exec -g portfolio_opt.mos RiskW=%%r ScenFile=%%d %FORMAT% SolFile=%RES_DIR%\!outBase!_opt_w%%r.sol Solver=%SOLVER% SolverPath=%SOLVER_PATH%
	)
)

echo:
echo Finish: %date% %time%

endlocal
