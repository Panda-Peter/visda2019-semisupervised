#!/usr/bin/env bash
#0917_eff5_real_clipart_no_aug_re_train.sh
#0917_eff5_real_painting_no_aug_re_train.sh
#0917_eff6_real_clipart_no_aug_re_train.sh
#0917_eff6_real_painting_no_aug_re_train.sh
#0917_eff7_real_clipart_no_aug_re_train.sh
#0917_eff7_real_painting_no_aug_re_train.sh
#0917_se101_real_clipart_no_aug_re_train.sh
#0917_se101_real_painting_no_aug_re_train.sh
#0917_se152_real_clipart_no_aug_re_train.sh
#0917_se152_real_painting_no_aug_re_train.sh
#0918_eff4_real_clipart_no_aug_re_train.sh
#0918_eff4_real_painting_no_aug_re_train.sh
#0918_inception_real_clipart_no_aug_re_train.sh
#0918_inception_real_painting_no_aug_re_train.sh
#
#source_only_models_0920-0917_eff5_real_clipart_no_aug_re_train_snapshot_netG_25.pth
#source_only_models_0920-0917_eff5_real_painting_no_aug_re_train_snapshot_netG_29.pth
#source_only_models_0920-0917_eff6_real_clipart_no_aug_re_train_snapshot_netG_28.pth
#source_only_models_0920-0917_eff6_real_painting_no_aug_re_train_snapshot_netG_28.pth
#source_only_models_0920-0917_eff7_real_clipart_no_aug_re_train_snapshot_netG_29.pth
#source_only_models_0920-0917_eff7_real_painting_no_aug_re_train_snapshot_netG_14.pth
#source_only_models_0920-0917_se101_real_clipart_no_aug_re_train_snapshot_netG_28.pth
#source_only_models_0920-0917_se101_real_painting_no_aug_re_train_snapshot_netG_29.pth
#source_only_models_0920-0917_se152_real_clipart_no_aug_re_train_snapshot_netG_29.pth
#source_only_models_0920-0917_se152_real_painting_no_aug_re_train_snapshot_netG_15.pth
#source_only_models_0920-0918_eff4_real_clipart_no_aug_re_train_snapshot_netG_29.pth
#source_only_models_0920-0918_eff4_real_painting_no_aug_re_train_snapshot_netG_29.pth
#source_only_models_0920-0918_inception_real_clipart_no_aug_re_train_snapshot_netG_29.pth
#source_only_models_0920-0918_inception_real_painting_no_aug_re_train_snapshot_netG_29.pth

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 evaluate_models.py --config ./configs/0917_eff5_real_clipart_no_aug_re_train.yml --netG_model_path experiments/full_set_models/source_only_models_0920-0917_eff5_real_clipart_no_aug_re_train_snapshot_netG_25.pth --output_path experiments/full_set_predictions/0917_eff5_real_clipart_no_aug_re_train.txt --multi_crop 1 --test_crop_size 224 --test_resize_size 256 --save_feature 1 --test_bs 64
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 evaluate_models.py --config ./configs/0917_eff5_real_painting_no_aug_re_train.yml --netG_model_path experiments/full_set_models/source_only_models_0920-0917_eff5_real_painting_no_aug_re_train_snapshot_netG_29.pth --output_path experiments/full_set_predictions/0917_eff5_real_painting_no_aug_re_train.txt --multi_crop 1 --test_crop_size 224 --test_resize_size 256 --save_feature 1 --test_bs 64
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 evaluate_models.py --config ./configs/0917_eff6_real_clipart_no_aug_re_train.yml --netG_model_path experiments/full_set_models/source_only_models_0920-0917_eff6_real_clipart_no_aug_re_train_snapshot_netG_28.pth --output_path experiments/full_set_predictions/0917_eff6_real_clipart_no_aug_re_train.txt --multi_crop 1 --test_crop_size 224 --test_resize_size 256 --save_feature 1 --test_bs 64
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 evaluate_models.py --config ./configs/0917_eff6_real_painting_no_aug_re_train.yml --netG_model_path experiments/full_set_models/source_only_models_0920-0917_eff6_real_painting_no_aug_re_train_snapshot_netG_28.pth --output_path experiments/full_set_predictions/0917_eff6_real_painting_no_aug_re_train.txt --multi_crop 1 --test_crop_size 224 --test_resize_size 256 --save_feature 1 --test_bs 64
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 evaluate_models.py --config ./configs/0917_eff7_real_clipart_no_aug_re_train.yml --netG_model_path experiments/full_set_models/source_only_models_0920-0917_eff7_real_clipart_no_aug_re_train_snapshot_netG_29.pth --output_path experiments/full_set_predictions/0917_eff7_real_clipart_no_aug_re_train.txt --multi_crop 1 --test_crop_size 224 --test_resize_size 256 --save_feature 1 --test_bs 64
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 evaluate_models.py --config ./configs/0917_eff7_real_painting_no_aug_re_train.yml --netG_model_path experiments/full_set_models/source_only_models_0920-0917_eff7_real_painting_no_aug_re_train_snapshot_netG_14.pth --output_path experiments/full_set_predictions/0917_eff7_real_painting_no_aug_re_train.txt --multi_crop 1 --test_crop_size 224 --test_resize_size 256 --save_feature 1 --test_bs 64
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 evaluate_models.py --config ./configs/0917_se101_real_clipart_no_aug_re_train.yml --netG_model_path experiments/full_set_models/source_only_models_0920-0917_se101_real_clipart_no_aug_re_train_snapshot_netG_28.pth --output_path experiments/full_set_predictions/0917_se101_real_clipart_no_aug_re_train.txt --multi_crop 1 --test_crop_size 224 --test_resize_size 256 --save_feature 1 --test_bs 64
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 evaluate_models.py --config ./configs/0917_se101_real_painting_no_aug_re_train.yml --netG_model_path experiments/full_set_models/source_only_models_0920-0917_se101_real_painting_no_aug_re_train_snapshot_netG_29.pth --output_path experiments/full_set_predictions/0917_se101_real_painting_no_aug_re_train.txt --multi_crop 1 --test_crop_size 224 --test_resize_size 256 --save_feature 1 --test_bs 64
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 evaluate_models.py --config ./configs/0917_se152_real_clipart_no_aug_re_train.yml --netG_model_path experiments/full_set_models/source_only_models_0920-0917_se152_real_clipart_no_aug_re_train_snapshot_netG_29.pth --output_path experiments/full_set_predictions/0917_se152_real_clipart_no_aug_re_train.txt --multi_crop 1 --test_crop_size 224 --test_resize_size 256 --save_feature 1 --test_bs 64
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 evaluate_models.py --config ./configs/0917_se152_real_painting_no_aug_re_train.yml --netG_model_path experiments/full_set_models/source_only_models_0920-0917_se152_real_painting_no_aug_re_train_snapshot_netG_15.pth --output_path experiments/full_set_predictions/0917_se152_real_painting_no_aug_re_train.txt --multi_crop 1 --test_crop_size 224 --test_resize_size 256 --save_feature 1 --test_bs 64
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 evaluate_models.py --config ./configs/0918_eff4_real_clipart_no_aug_re_train.yml --netG_model_path experiments/full_set_models/source_only_models_0920-0918_eff4_real_clipart_no_aug_re_train_snapshot_netG_29.pth --output_path experiments/full_set_predictions/0918_eff4_real_clipart_no_aug_re_train.txt --multi_crop 1 --test_crop_size 224 --test_resize_size 256 --save_feature 1 --test_bs 64
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 evaluate_models.py --config ./configs/0918_eff4_real_painting_no_aug_re_train.yml --netG_model_path experiments/full_set_models/source_only_models_0920-0918_eff4_real_painting_no_aug_re_train_snapshot_netG_29.pth --output_path experiments/full_set_predictions/0918_eff4_real_painting_no_aug_re_train.txt --multi_crop 1 --test_crop_size 224 --test_resize_size 256 --save_feature 1 --test_bs 64
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 evaluate_models.py --config ./configs/0918_inception_real_clipart_no_aug_re_train.yml --netG_model_path experiments/full_set_models/source_only_models_0920-0918_inception_real_clipart_no_aug_re_train_snapshot_netG_29.pth --output_path experiments/full_set_predictions/0918_inception_real_clipart_no_aug_re_train.txt --multi_crop 1 --test_crop_size 224 --test_resize_size 256 --save_feature 1 --test_bs 64
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 evaluate_models.py --config ./configs/0918_inception_real_painting_no_aug_re_train.yml --netG_model_path experiments/full_set_models/source_only_models_0920-0918_inception_real_painting_no_aug_re_train_snapshot_netG_29.pth --output_path experiments/full_set_predictions/0918_inception_real_painting_no_aug_re_train.txt --multi_crop 1 --test_crop_size 224 --test_resize_size 256 --save_feature 1 --test_bs 64
