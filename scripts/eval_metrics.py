import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
from scipy.stats import wasserstein_distance
from sklearn.neighbors import KernelDensity
import torch
import os
import pandas as pd

from guided_diffusion.mlp_model import MLPDiff
from guided_diffusion.gaussian_diffusion import create_gaussian_diffusion
from guided_diffusion.dist_util import dev
from datasets.rings2d import TwoRingGaussians

def compute_wasserstein(real, fake):
    wd_x = wasserstein_distance(real[:, 0], fake[:, 0])
    wd_y = wasserstein_distance(real[:, 1], fake[:, 1])
    return wd_x + wd_y

def compute_mode_recovery(samples, eps=0.3, min_samples=20):
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(samples)
    labels = clustering.labels_
    unique_modes = len(set(labels)) - (1 if -1 in labels else 0)
    return unique_modes

def compute_precision_recall(real, fake, k=5):
    r2r = pairwise_distances(real, real)
    r2f = pairwise_distances(real, fake)
    f2r = pairwise_distances(fake, real)
    f2f = pairwise_distances(fake, fake)
    r_k = np.sort(r2r, axis=1)[:, k]
    f_k = np.sort(f2f, axis=1)[:, k]
    precision = (np.min(f2r, axis=1) < r_k).mean()
    recall = (np.min(r2f, axis=1) < f_k).mean()
    return precision, recall

def plot_kde(samples, title, fname):
    kde = KernelDensity(bandwidth=0.2).fit(samples)
    x = np.linspace(-6, 6, 300)
    y = np.linspace(-6, 6, 300)
    xx, yy = np.meshgrid(x, y)
    xy = np.vstack([xx.ravel(), yy.ravel()]).T
    z = np.exp(kde.score_samples(xy)).reshape(xx.shape)
    plt.figure(figsize=(4, 4))
    plt.contourf(xx, yy, z, levels=50)
    plt.scatter(samples[:, 0], samples[:, 1], s=2, alpha=0.5, c='white')
    plt.title(title)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()

@torch.no_grad()
def sample_model(model, diffusion, num_points=1000):
    return diffusion.p_sample_loop(model, shape=(num_points, 2), clip_denoised=False).cpu().numpy()

def load_model(ckpt_path, width):
    model = MLPDiff(model_channels=width, n_layers=3).to(dev())
    model.load_state_dict(torch.load(ckpt_path, map_location=dev()))
    model.eval()
    return model

def generate_real_samples(n_samples=1000, r1=2.0, r2=4.0, std=0.05):
    dataset = TwoRingGaussians(n_samples=n_samples, r1=r1, r2=r2, std=std)
    return np.stack([dataset[i][0].numpy() for i in range(len(dataset))])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", type=str, help="Path to .npy file of real samples")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--steps", type=int, default=1000, help="Number of diffusion steps")
    parser.add_argument("--width", type=int, default=256, help="Width of hidden layers in model")
    parser.add_argument("--n_gen", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument("--n_real", type=int, default=1000, help="Number of real samples to generate if --real not provided")
    parser.add_argument("--outdir", type=str, default="eval_results", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if args.real:
        real = np.load(args.real)
    else:
        real = generate_real_samples(n_samples=args.n_real)
        np.save(os.path.join(args.outdir, "real.npy"), real)

    diffusion = create_gaussian_diffusion(steps=args.steps)
    model = load_model(args.ckpt, width=args.width)
    fake = sample_model(model, diffusion, num_points=args.n_gen)

    wd = compute_wasserstein(real, fake)
    modes = compute_mode_recovery(fake)
    precision, recall = compute_precision_recall(real, fake)

    plot_kde(real, "Real KDE", os.path.join(args.outdir, "real_kde.png"))
    plot_kde(fake, "Fake KDE", os.path.join(args.outdir, "fake_kde.png"))

    summary = {
        "checkpoint": os.path.basename(args.ckpt),
        "Wasserstein Distance": wd,
        "Recovered Modes": modes,
        "Precision (k=5)": precision,
        "Recall (k=5)": recall
    }

    summary_txt_path = os.path.join(args.outdir, "metrics.txt")
    with open(summary_txt_path, "w") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    summary_csv_path = os.path.join(args.outdir, "metrics.csv")
    df = pd.DataFrame([summary])
    df.to_csv(summary_csv_path, index=False)

    print("Evaluation complete. Metrics saved to:", summary_txt_path)

if __name__ == "__main__":
    main()

