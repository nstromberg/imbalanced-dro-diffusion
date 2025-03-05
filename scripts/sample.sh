#!/bin/bash

python -V
pip -V
pip list
pwd

workdir=${workdir}
ALL_STEPS=(450000)
for STEPS in ${ALL_STEPS[@]}
do
export CHECKPOINT_PATH=${workdir}/ema_0.9999_${STEPS}.pt

sample_nfes=(50 20 10 8 5)
for i in ${!sample_nfes[@]}
do
    nfe=${sample_nfes[${i}]}
    CUDA_VISIBLE_DEVICES=0,1,2,3 mpiexec -n 4 python scripts/image_sample.py \
        --image_size 32 \
        --model_path ${CHECKPOINT_PATH} --ckpt_step ${STEPS} --sample_dir ${workdir}/iddpm \
        --num_channels 128 --num_head_channels 32 --num_res_blocks 3 --attention_resolutions 16,8 \
        --resblock_updown True --use_new_attention_order True --learn_sigma True --dropout 0.3 \
        --diffusion_steps 1000 --noise_schedule cosine --use_scale_shift_norm True \
        --use_fp16=True --batch_size 256 --num_samples 50000 --timestep_respacing ${nfe} --use_ddim False \
        --eps_scaler 1.0
done

sample_nfes=(50 20 10 8 5)
for i in ${!sample_nfes[@]}
do
    nfe=${sample_nfes[${i}]}
    CUDA_VISIBLE_DEVICES=0,1,2,3 mpiexec -n 4 python scripts/image_sample.py \
        --image_size 32 \
        --model_path ${CHECKPOINT_PATH} --ckpt_step ${STEPS} --sample_dir ${workdir}/es \
        --num_channels 128 --num_head_channels 32 --num_res_blocks 3 --attention_resolutions 16,8 \
        --resblock_updown True --use_new_attention_order True --learn_sigma True --dropout 0.3 \
        --diffusion_steps 1000 --noise_schedule cosine --use_scale_shift_norm True \
        --use_fp16=True --batch_size 256 --num_samples 50000 --timestep_respacing ${nfe} --use_ddim False \
        --eps_scaler 1.017
done

sample_nfes=(ddim50 ddim20 ddim10 ddim8 ddim5)
for i in ${!sample_nfes[@]}
do
    nfe=${sample_nfes[${i}]}
    CUDA_VISIBLE_DEVICES=0,1,2,3 mpiexec -n 4 python scripts/image_sample.py \
        --image_size 32 \
        --model_path ${CHECKPOINT_PATH} --ckpt_step ${STEPS} --sample_dir ${workdir}/ddim \
        --num_channels 128 --num_head_channels 32 --num_res_blocks 3 --attention_resolutions 16,8 \
        --resblock_updown True --use_new_attention_order True --learn_sigma True --dropout 0.3 \
        --diffusion_steps 1000 --noise_schedule cosine --use_scale_shift_norm True \
        --use_fp16=True --batch_size 256 --num_samples 50000 --timestep_respacing ${nfe} --use_ddim True
done

sample_nfes=(5 8 10 20 50)
for i in ${!sample_nfes[@]}
do
    nfe=${sample_nfes[${i}]}
    CUDA_VISIBLE_DEVICES=0,1,2,3 mpiexec -n 4 python scripts/image_dpm_solver_sample.py \
        --image_size 32 \
        --model_path ${CHECKPOINT_PATH} --ckpt_step ${STEPS} --sample_dir ${workdir}/ \
        --num_channels 128 --num_head_channels 32 --num_res_blocks 3 --attention_resolutions 16,8 \
        --resblock_updown True --use_new_attention_order True --learn_sigma True --dropout 0.3 \
        --diffusion_steps 1000 --noise_schedule cosine --use_scale_shift_norm True \
        --use_fp16=True --batch_size 256 --num_samples 50000 --steps ${nfe}
done
done