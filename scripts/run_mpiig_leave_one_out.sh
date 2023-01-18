#!/bin/bash

for i in {0..14}
do 
    python3 train.py \
        --config-name=mpiig_leave_one_out_ranger3 \
        dataset.hdf5_name=mpiig.hdf5 \
        run_options.leave_out.idx_val=$i \
        run_options.leave_out.idx_test=$i \
        run_options.new_norm=False \
        train_options.max_epochs=15 \
        > mpiig_run_idx_$i.out
done