"""
Train a diffusion model on images.
"""

import argparse
import torch.distributed as dist

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
    adv_defaults,
    set_seed,
)
from guided_diffusion.adv_train_util import AdvTrainLoop


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    rank = dist.get_rank()
    seed = args.global_seed * dist.get_world_size() + rank
    set_seed(seed)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    logger.log(args)
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    if args.resume_checkpoint != "":
        logger.log(f"Train model from {args.resume_checkpoint}.")
    elif args.load_from_ema_ckpt != "":
        logger.log(f'Train model from EMA ckpt {args.load_from_ema_ckpt}.')
    else:
        logger.log("Train model from scratch.")
    
    if args.use_fp16:
        model.convert_to_fp16()
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log(f"creating data loader from {args.data_dir}...")
    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )

    logger.log("training...")
    AdvTrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        adv_training_type=args.adv_training_type,  # [freelb, fast_adv, ...]
        adv_norm_type=args.adv_norm_type,      # [l1, l2, linf]
        adv_steps=args.adv_steps,
        adv_lr=args.adv_lr,
        adv_max_norm=args.adv_max_norm,
        adv_init_mag=args.adv_init_mag,
        adv_target_type=args.adv_target_type,
        load_from_ema_ckpt=args.load_from_ema_ckpt,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=100,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        global_seed=42,
        load_from_ema_ckpt="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
