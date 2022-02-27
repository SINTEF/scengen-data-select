@echo off

:: generate scenario sets

setlocal enabledelayedexpansion

set DATA="prices\prices_1w_all_*.csv"
set RES_DIR="scens"
set FORMAT=--col-sep tab --index-col date
set METHODS=sampling k-means Wasserstein optimization
set SCENS=10 20 50 100
set NTREES=25
set EQUIPROB=0
set LOG_FILE=gen_scen.log

if not exist %RES_DIR% mkdir %RES_DIR%

echo Start: %date% %time%
echo:

for %%d in (%DATA%) do (
	set inFile=%%~nd
	set outBase=%RES_DIR%\!inFile:prices=scen!
	set variants=default
	for %%m in (%METHODS%) do (
		set method=%%m
		set mNtrees=%NTREES%
		if %%m==sampling (
			set variants=500
		) else if %%m==sampling-WD (
			set variants=500 10000
		) else if %%m==optimization (
			set variants=300, 1800
			set mNtrees=1
		)
		for %%v in (!variants!) do (
			if %%m==sampling (
				set M_OPT=--n-samples %%v
				set method=%%m-%%v
			) else if %%m==sampling-WD (
				set M_OPT=--n-samples %%v
				set method=%%m-%%v
			) else if %%m==optimization (
				set M_OPT=--max-time %%v
				set method=%%m-%%v
			) else (
				set M_OPT=
			)
			if %EQUIPROB%==1 (
				set M_OPT=!M_OPT! --equiprob
			)
			for %%s in (%SCENS%) do (
				for /l %%t in (1,1,!mNtrees!) do (
					if not exist !outBase!_!method!_%%ss_%%t.csv (
						python ..\..\scengen_select.py -d %%d %FORMAT% -m %%m !M_OPT! -s %%s -o !outBase!_!method!_%%ss_%%t | tee -a -i %LOG_FILE%
					)
				)
			)
		)
	)
)

echo:
echo Finish: %date% %time%

endlocal
