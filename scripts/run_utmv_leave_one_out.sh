#!/bin/bash

for i in {0..2}
do 
    python3 train.py \
    --config-name=mpiig_leave_one_out_ranger3 \
    dataset.hdf5_name=utmvfromsynth.hdf5 \
    +run_options.k_fold_validation=3 \
    +run_options.k_fold_validation_idx=$i \
    run_options.new_norm=True \
    > utmv_run_3fold_idx_$i.out
done