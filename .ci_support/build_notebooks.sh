#!/bin/bash
for f in $(cat .ci_support/exclude); do 
    rm "notebooks/$f";     
done;
# execute notebooks
i=0;
cd notebooks
for notebook in $(ls ./fenics_*.ipynb); do
    papermill ${notebook} ${notebook%.*}-out.${notebook##*.} -k "python3" || i=$((i+1));
done;

# push error to next level
if [ $i -gt 0 ]; then
    exit 1;
fi;