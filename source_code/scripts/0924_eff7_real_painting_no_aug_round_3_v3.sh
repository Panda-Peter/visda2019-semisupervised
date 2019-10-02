export NCCL_LL_THRESHOLD=0
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main.py --config ./configs/0924_eff7_real_painting_no_aug_round_3_v3.yml
