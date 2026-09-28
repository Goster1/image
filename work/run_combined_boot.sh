#!/bin/bash
# bootstrap + finalize only (the fit in cache/combined_fit.json is kept): 4 parallel chunks of 25 replicates
cd "$(dirname "$0")"
rm -f cache/combined_boot_*.json
for s in 11 22 33 44; do OMP_NUM_THREADS=1 python3 50_combined.py boot $s 25 > cache/log_50_boot_$s.txt 2>&1 & done
wait
python3 50_combined.py finalize > cache/log_50_finalize.txt 2>&1
echo ALLDONE >> cache/log_50_finalize.txt
