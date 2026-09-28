#!/bin/bash
# fit (variants, profile), then 4 parallel bootstrap chunks (4 x 25), then finalize
cd "$(dirname "$0")"
python3 50_combined.py fit > cache/log_50_fit.txt 2>&1 || exit 1
rm -f cache/combined_boot_*.json
for s in 11 22 33 44; do python3 50_combined.py boot $s 25 > cache/log_50_boot_$s.txt 2>&1 & done
wait
python3 50_combined.py finalize > cache/log_50_finalize.txt 2>&1
echo ALLDONE >> cache/log_50_finalize.txt
