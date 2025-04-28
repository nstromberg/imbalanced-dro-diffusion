import argparse, torch
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import create_gaussian_diffusion
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.adv_train_util import AdvTrainLoop     # reuse AT‑Diff loop
from guided_diffusion.mlp_model import MLPDiff
from datasets.rings2d import load_data_points

def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    logger.configure(
        dir='logs/mlp_rings',
        format_strs=['log','stdout'],
        log_suffix='rings'
    )
    
    # 1. model & diffusion --------------------------------------------------
    model = MLPDiff(model_channels=args.width).to(dist_util.dev())
    diffusion = create_gaussian_diffusion(      # identical helper used by scripts/image_train.py
        steps=args.steps,
        learn_sigma=False,
        noise_schedule=args.noise_schedule,
        use_kl=False,
    )
    if args.use_fp16:
        model.convert_to_fp16()
    
    # 2. data loader --------------------------------------------------------
    data = load_data_points(batch_size=args.batch_size,
    n_samples=args.n_samples,
    r1=args.r1, r2=args.r2, std=args.std)
    print('Data loaded')
    
    # 3. train --------------------------------------------------------------
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)
    AdvTrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=-1,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=100,
        save_interval=10_000,
        resume_checkpoint="",
        use_fp16=args.use_fp16,
        fp16_scale_growth=1e-3,
        schedule_sampler=schedule_sampler,
        weight_decay=0.0,
        lr_anneal_steps=0,
        adv_training_type=None,            # turn off adversarial part for first experiments
        max_steps = args.max_steps,
    ).run_loop()
    
def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--noise_schedule", type=str, default="cosine")
    parser.add_argument("--schedule_sampler", type=str, default="uniform")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--ema_rate", type=str, default="0.999")
    parser.add_argument("--use_fp16", action="store_true")
    # ring‑specific
    parser.add_argument("--n_samples", type=int, default=100_000)
    parser.add_argument("--r1", type=float, default=2.0)
    parser.add_argument("--r2", type=float, default=4.0)
    parser.add_argument("--std", type=float, default=0.05)
    parser.add_argument("--width", type=int, default=256)
    return parser
    
if __name__ == "__main__":
    main()
    
