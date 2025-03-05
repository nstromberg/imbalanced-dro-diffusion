"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
import math


from tqdm import tqdm
from PIL import Image


from guided_diffusion.gaussian_diffusion import get_named_beta_schedule
from torch.cuda.amp import autocast
from guided_diffusion.dpm_solver import NoiseScheduleVP, model_wrapper, DPM_Solver

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    IMAGENET_NUM_CLASSES,
    CIFAR10_NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
    set_seed,
)


def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path

def main():
    args = create_argparser().parse_args()
    
    dist_util.setup_dist()
    logger.configure()

    rank = dist.get_rank()
    device = rank % th.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    set_seed(seed)
    th.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(device)
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    
    
    folder_name = f"ckpt{args.ckpt_step}-steps{args.steps}"    
    sample_folder_dir = f"{args.sample_dir}/dpm-solver/{folder_name}"
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()
    

    all_images = []
    n = args.batch_size
    global_batch_size = n * dist.get_world_size()
    total_samples = int(math.ceil(args.num_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0
    betas = th.from_numpy(get_named_beta_schedule("cosine", 1000))
    ns = NoiseScheduleVP(schedule='discrete', betas = betas)    
    
    NUM_CLASSES = CIFAR10_NUM_CLASSES if args.image_size == 32 else IMAGENET_NUM_CLASSES
    with autocast(dtype = th.float16):
        for _ in pbar:
            model_kwargs = {}
            z = th.randn(n, 3, args.image_size, args.image_size, device=device)

            if args.class_cond:
                y = th.randint(0, NUM_CLASSES, (n,), device=device)

            model_fn = model_wrapper(
                model,
                ns,
                model_type = "noise",
                guidance_type = "classifier-free" if args.class_cond else "uncond",
                condition = y if args.class_cond else None,
                unconditional_condition = None,
                guidance_scale = 1.
            )
            sampler = DPM_Solver(model_fn, ns, algorithm_type='dpmsolver++')
            samples = sampler.sample(x = z, steps=args.steps, skip_type="logSNR", t_start=0.992)

            samples = ((samples + 1) * 127.5).clamp(0, 255)
            samples = samples.permute(0, 2, 3, 1)
            samples = samples.contiguous().to("cpu", dtype=th.uint8).numpy()

            # Save samples to disk as individual .png files
            for i, sample in enumerate(samples):
                index = i * dist.get_world_size() + rank + total
                Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
            total += global_batch_size  




# Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    if dist.get_rank() == 0:
        create_npz_from_sample_folder(sample_folder_dir, args.num_samples)
        print("Done.")
    dist.barrier()
    dist.destroy_process_group()



def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        global_seed=42,
        ckpt_step=0,
        sample_dir="",
        steps=20,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
