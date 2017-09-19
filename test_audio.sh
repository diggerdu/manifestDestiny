python test_debug.py \
 --pathA "/home/diggerdu/dataset/men-dev/A"\
 --pathB "/home/diggerdu/dataset/men-dev/B"\
 --name audio-res9-lr:0.01 --model pix2pix --which_model_netG resnet_6blocks \
 --ngf 64 \
 --input_nc 1 --output_nc 1  \
 --which_direction AtoB --nThreads 1 \
 --size 512*256 --gpu_ids 2 --batchSize 1  --how_many 6 \
 --hop 256 \
 --split_hop 0 \
