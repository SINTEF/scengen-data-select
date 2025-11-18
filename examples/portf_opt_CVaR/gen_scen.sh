#!/bin/bash

# generate scenario sets

DATA="prices/prices_1w_all_*.csv"
RES_DIR="scens"
FORMAT="--col-sep tab --index-col date"
METHODS=(sampling k-means Wasserstein optimization)
SCENS=(10 20 50 100)
NTREES=25
EQUIPROB=0
LOG_FILE=gen_scen.log

if [ ! -d "$RES_DIR" ]; then
	mkdir "$RES_DIR"
fi

echo "Start: $(date)"
echo

for d in $DATA; do
	inFile=$(basename "$d" .csv)
	outBase="$RES_DIR/${inFile/prices/scen}"
	for m in "${METHODS[@]}"; do
		method="$m"
		mNtrees=$NTREES
		if [ "$m" = "sampling" ]; then
			variants=(500)
		elif [ "$m" = "sampling-WD" ]; then
			variants=(500 10000)
		elif [ "$m" = "optimization" ]; then
			variants=(300 1800)
			mNtrees=1
		else
			variants=(default)
		fi
		
		for v in "${variants[@]}"; do
			if [ "$m" = "sampling" ]; then
				M_OPT="--n-samples $v"
				method="$m-$v"
			elif [ "$m" = "sampling-WD" ]; then
				M_OPT="--n-samples $v"
				method="$m-$v"
			elif [ "$m" = "optimization" ]; then
				M_OPT="--max-time $v"
				method="$m-$v"
			else
				M_OPT=""
			fi
			
			if [ $EQUIPROB -eq 1 ]; then
				M_OPT="$M_OPT --equiprob"
			fi
			
			for s in "${SCENS[@]}"; do
				for t in $(seq 1 $mNtrees); do
					if [ ! -f "${outBase}_${method}_${s}s_${t}.csv" ]; then
						python ../../scengen_select.py -d "$d" $FORMAT -m "$m" $M_OPT -s "$s" -o "${outBase}_${method}_${s}s_${t}" | tee -a "$LOG_FILE"
					fi
				done
			done
		done
	done
done

echo
echo "Finish: $(date)"
