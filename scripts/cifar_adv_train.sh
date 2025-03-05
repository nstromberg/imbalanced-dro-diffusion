#!/bin/bash

python -V
pip -V
pip list
pwd

adv_type="fast"
norm_type="l2"
adv_steps=3
adv_lr=0.1
max_norm=0.1
init_mag=0.0
adv_target_type='delta_detach'
resume_checkpoint="results/cifar_adm/model250000.pt"
export OPENAI_LOGDIR=results/cifar_adm_from_250K_adv_${adv_type}_${norm_type}_${adv_steps}_${adv_lr}_${max_norm}_${init_mag}_${adv_target_type}/

CUDA_VISIBLE_DEVICES=0,1,2,3 mpiexec -n 4 python scripts/adv_image_train.py \
    --data_dir data/cifar_train --image_size 32 \
    --use_fp16 True --num_channels 128 --num_head_channels 32 --num_res_blocks 3 \
    --attention_resolutions 16,8 --resblock_updown True --use_new_attention_order True \
    --learn_sigma True --dropout 0.3 --diffusion_steps 1000 --noise_schedule cosine --use_scale_shift_norm True \
    --rescale_learned_sigmas True --lr 1e-4 --schedule_sampler loss-second-moment --batch_size 32 \
    --adv_training_type ${adv_type} --adv_norm_type ${norm_type} \
    --adv_steps ${adv_steps} --adv_lr ${adv_lr} --adv_max_norm ${max_norm} --adv_init_mag ${init_mag} \
    --adv_target_type ${adv_target_type} \
    --resume_checkpoint ${resume_checkpoint}
