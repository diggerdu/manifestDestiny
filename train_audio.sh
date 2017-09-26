#!/usr/bin/env bash
python train_debug.py \
 --PathClean "/home/diggerdu/dataset/men/clean" \
 --PathNoise "/home/diggerdu/dataset/men/noise" \
 --snr 0 \
 --name e2e --model pix2pix --which_model_netG wide_resnet_3blocks \
 --ngf 32 \
 --which_direction AtoB --lambda_A 100 --no_lsgan --nThreads 6 \
 --input_nc 1 --output_nc 1 \
 --nfft 256 --hop 128 --nFrames 12 --batchSize  6\
 --split_hop 0 \
 --niter 10000 --niter_decay 30 \
 --lr 0.001 \
 --gpu_ids 1
#  --serial_batches
