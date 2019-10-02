#!/usr/bin/env bash
# 0920_eff4_real_clipart_no_aug_round_1_v2
# 0920_eff4_real_painting_no_aug_round_1_v2
# 0920_eff5_real_clipart_no_aug_round_1_v2
# 0920_eff5_real_painting_no_aug_round_1_v2
# 0920_eff6_real_clipart_no_aug_round_1_v2
# 0920_eff6_real_painting_no_aug_round_1_v2
# 0920_eff7_real_clipart_no_aug_round_1_v2
# 0920_eff7_real_painting_no_aug_round_1_v2
# 0920_inception_real_clipart_no_aug_round_1_v2
# 0920_inception_real_painting_no_aug_round_1_v2
# 0920_se101_real_clipart_no_aug_round_1_v2
# 0920_se101_real_painting_no_aug_round_1_v2
# 0920_se152_real_clipart_no_aug_round_1_v2
# 0920_se152_real_painting_no_aug_round_1_v2

# pseudo_label_round_1_0922_v2-models-0920_eff4_real_clipart_no_aug_round_1_v2_snapshot_netG_11.pth
# pseudo_label_round_1_0922_v2-models-0920_eff4_real_painting_no_aug_round_1_v2_snapshot_netG_10.pth
# pseudo_label_round_1_0922_v2-models-0920_eff5_real_clipart_no_aug_round_1_v2_snapshot_netG_11.pth
# pseudo_label_round_1_0922_v2-models-0920_eff5_real_painting_no_aug_round_1_v2_snapshot_netG_15.pth
# pseudo_label_round_1_0922_v2-models-0920_eff6_real_clipart_no_aug_round_1_v2_snapshot_netG_9.pth
# pseudo_label_round_1_0922_v2-models-0920_eff6_real_painting_no_aug_round_1_v2_snapshot_netG_8.pth
# pseudo_label_round_1_0922_v2-models-0920_eff7_real_clipart_no_aug_round_1_v2_snapshot_netG_5.pth
# pseudo_label_round_1_0922_v2-models-0920_eff7_real_painting_no_aug_round_1_v2_snapshot_netG_4.pth
# pseudo_label_round_1_0922_v2-models-0920_inception_real_clipart_no_aug_round_1_v2_snapshot_netG_10.pth
# pseudo_label_round_1_0922_v2-models-0920_inception_real_painting_no_aug_round_1_v2_snapshot_netG_9.pth
# pseudo_label_round_1_0922_v2-models-0920_se101_real_clipart_no_aug_round_1_v2_snapshot_netG_11.pth
# pseudo_label_round_1_0922_v2-models-0920_se101_real_painting_no_aug_round_1_v2_snapshot_netG_10.pth
# pseudo_label_round_1_0922_v2-models-0920_se152_real_clipart_no_aug_round_1_v2_snapshot_netG_5.pth
# pseudo_label_round_1_0922_v2-models-0920_se152_real_painting_no_aug_round_1_v2_snapshot_netG_5.pth

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 evaluate_models.py --config ./configs/0920_eff4_real_clipart_no_aug_round_1_v2.yml --netG_model_path experiments/full_set_models/pseudo_label_round_1_0922_v2-models-0920_eff4_real_clipart_no_aug_round_1_v2_snapshot_netG_11.pth --output_path experiments/full_set_predictions/0920_eff4_real_clipart_no_aug_round_1_v2.txt --multi_crop 1 --test_crop_size 224 --test_resize_size 256 --save_feature 1 --test_bs 64
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 evaluate_models.py --config ./configs/0920_eff4_real_painting_no_aug_round_1_v2.yml --netG_model_path experiments/full_set_models/pseudo_label_round_1_0922_v2-models-0920_eff4_real_painting_no_aug_round_1_v2_snapshot_netG_10.pth --output_path experiments/full_set_predictions/0920_eff4_real_painting_no_aug_round_1_v2.txt --multi_crop 1 --test_crop_size 224 --test_resize_size 256 --save_feature 1 --test_bs 64
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 evaluate_models.py --config ./configs/0920_eff5_real_clipart_no_aug_round_1_v2.yml --netG_model_path experiments/full_set_models/pseudo_label_round_1_0922_v2-models-0920_eff5_real_clipart_no_aug_round_1_v2_snapshot_netG_11.pth --output_path experiments/full_set_predictions/0920_eff5_real_clipart_no_aug_round_1_v2.txt --multi_crop 1 --test_crop_size 224 --test_resize_size 256 --save_feature 1 --test_bs 64
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 evaluate_models.py --config ./configs/0920_eff5_real_painting_no_aug_round_1_v2.yml --netG_model_path experiments/full_set_models/pseudo_label_round_1_0922_v2-models-0920_eff5_real_painting_no_aug_round_1_v2_snapshot_netG_15.pth --output_path experiments/full_set_predictions/0920_eff5_real_painting_no_aug_round_1_v2.txt --multi_crop 1 --test_crop_size 224 --test_resize_size 256 --save_feature 1 --test_bs 64
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 evaluate_models.py --config ./configs/0920_eff6_real_clipart_no_aug_round_1_v2.yml --netG_model_path experiments/full_set_models/pseudo_label_round_1_0922_v2-models-0920_eff6_real_clipart_no_aug_round_1_v2_snapshot_netG_9.pth --output_path experiments/full_set_predictions/0920_eff6_real_clipart_no_aug_round_1_v2.txt --multi_crop 1 --test_crop_size 224 --test_resize_size 256 --save_feature 1 --test_bs 64
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 evaluate_models.py --config ./configs/0920_eff6_real_painting_no_aug_round_1_v2.yml --netG_model_path experiments/full_set_models/pseudo_label_round_1_0922_v2-models-0920_eff6_real_painting_no_aug_round_1_v2_snapshot_netG_8.pth --output_path experiments/full_set_predictions/0920_eff6_real_painting_no_aug_round_1_v2.txt --multi_crop 1 --test_crop_size 224 --test_resize_size 256 --save_feature 1 --test_bs 64
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 evaluate_models.py --config ./configs/0920_eff7_real_clipart_no_aug_round_1_v2.yml --netG_model_path experiments/full_set_models/pseudo_label_round_1_0922_v2-models-0920_eff7_real_clipart_no_aug_round_1_v2_snapshot_netG_5.pth --output_path experiments/full_set_predictions/0920_eff7_real_clipart_no_aug_round_1_v2.txt --multi_crop 1 --test_crop_size 224 --test_resize_size 256 --save_feature 1 --test_bs 64
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 evaluate_models.py --config ./configs/0920_eff7_real_painting_no_aug_round_1_v2.yml --netG_model_path experiments/full_set_models/pseudo_label_round_1_0922_v2-models-0920_eff7_real_painting_no_aug_round_1_v2_snapshot_netG_4.pth --output_path experiments/full_set_predictions/0920_eff7_real_painting_no_aug_round_1_v2.txt --multi_crop 1 --test_crop_size 224 --test_resize_size 256 --save_feature 1 --test_bs 64
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 evaluate_models.py --config ./configs/0920_inception_real_clipart_no_aug_round_1_v2.yml --netG_model_path experiments/full_set_models/pseudo_label_round_1_0922_v2-models-0920_inception_real_clipart_no_aug_round_1_v2_snapshot_netG_10.pth --output_path experiments/full_set_predictions/0920_inception_real_clipart_no_aug_round_1_v2.txt --multi_crop 1 --test_crop_size 224 --test_resize_size 256 --save_feature 1 --test_bs 64
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 evaluate_models.py --config ./configs/0920_inception_real_painting_no_aug_round_1_v2.yml --netG_model_path experiments/full_set_models/pseudo_label_round_1_0922_v2-models-0920_inception_real_painting_no_aug_round_1_v2_snapshot_netG_9.pth --output_path experiments/full_set_predictions/0920_inception_real_painting_no_aug_round_1_v2.txt --multi_crop 1 --test_crop_size 224 --test_resize_size 256 --save_feature 1 --test_bs 64
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 evaluate_models.py --config ./configs/0920_se101_real_clipart_no_aug_round_1_v2.yml --netG_model_path experiments/full_set_models/pseudo_label_round_1_0922_v2-models-0920_se101_real_clipart_no_aug_round_1_v2_snapshot_netG_11.pth --output_path experiments/full_set_predictions/0920_se101_real_clipart_no_aug_round_1_v2.txt --multi_crop 1 --test_crop_size 224 --test_resize_size 256 --save_feature 1 --test_bs 64
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 evaluate_models.py --config ./configs/0920_se101_real_painting_no_aug_round_1_v2.yml --netG_model_path experiments/full_set_models/pseudo_label_round_1_0922_v2-models-0920_se101_real_painting_no_aug_round_1_v2_snapshot_netG_10.pth --output_path experiments/full_set_predictions/0920_se101_real_painting_no_aug_round_1_v2.txt --multi_crop 1 --test_crop_size 224 --test_resize_size 256 --save_feature 1 --test_bs 64
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 evaluate_models.py --config ./configs/0920_se152_real_clipart_no_aug_round_1_v2.yml --netG_model_path experiments/full_set_models/pseudo_label_round_1_0922_v2-models-0920_se152_real_clipart_no_aug_round_1_v2_snapshot_netG_5.pth --output_path experiments/full_set_predictions/0920_se152_real_clipart_no_aug_round_1_v2.txt --multi_crop 1 --test_crop_size 224 --test_resize_size 256 --save_feature 1 --test_bs 64
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 evaluate_models.py --config ./configs/0920_se152_real_painting_no_aug_round_1_v2.yml --netG_model_path experiments/full_set_models/pseudo_label_round_1_0922_v2-models-0920_se152_real_painting_no_aug_round_1_v2_snapshot_netG_5.pth --output_path experiments/full_set_predictions/0920_se152_real_painting_no_aug_round_1_v2.txt --multi_crop 1 --test_crop_size 224 --test_resize_size 256 --save_feature 1 --test_bs 64
