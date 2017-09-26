#!/usr/bin/env zsh
python test_debug.py \
 --PathClean "/home/diggerdu/dataset/men/clean" \
 --PathNoise "/home/diggerdu/dataset/men/noise" \
 --snr 0 \
 --name e2e --model test \
 --ngf 32 \
 --which_direction AtoB --nThreads 1\
 --input_nc 1 --output_nc 1 \
 --nfft 256 --hop 128 --nFrames 12 --batchSize  6\
 --split_hop 0 \
 --gpu_ids 1
#  --serial_batches
