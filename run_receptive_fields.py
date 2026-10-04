"""Experiment 3: gradient-saliency 'receptive fields' of tree nodes at different depths.
Usage: python run_receptive_fields.py --ckpt results/overfit/model.pt"""
import argparse, os
from argparse import Namespace
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tree_nn.cli import make_data
from tree_nn.models import load_tree
from tree_nn.train import get_device
from tree_nn.analysis import node_saliency, concentration, class_response

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--out", default="results/receptive_fields")
p.add_argument("--num-images", type=int, default=6)
p.add_argument("--stat-images", type=int, default=200)
p.add_argument("--smooth", type=int, default=10, help="SmoothGrad samples (1 = plain gradient)")
p.add_argument("--nodes", nargs="*", default=None, help="node paths, e.g. root root/0 root/0/1")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
dev = get_device()
model, ckpt = load_tree(a.ckpt, dev)
model.eval()
args = Namespace(**ckpt["args"])
data = make_data(args, dev)

# default: root + first child at every depth + the last leaf
if a.nodes is None:
    a.nodes = ["root"] + ["/".join(["root"] + ["0"] * d) for d in range(1, model.cfg.depth + 1)]
    leaves = [n.path for n in model.nodes() if n.depth == model.cfg.depth]
    if leaves[-1] not in a.nodes: a.nodes.append(leaves[-1])

x, y = data.xte[:a.num_images], data.yte[:a.num_images]
unnorm = lambda t: (t * data.std + data.mean).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
fig, ax = plt.subplots(len(a.nodes) + 1, a.num_images, figsize=(2 * a.num_images, 2 * (len(a.nodes) + 1)))
for j in range(a.num_images):
    ax[0, j].imshow(unnorm(x[j:j + 1])); ax[0, j].set_title(data.class_names[y[j]], fontsize=8)
for r, path in enumerate(a.nodes, 1):
    sal = node_saliency(model, x, path, n_samples=a.smooth).cpu().numpy()
    for j in range(a.num_images):
        ax[r, j].imshow(unnorm(x[j:j + 1])); ax[r, j].imshow(sal[j], cmap="jet", alpha=0.5)
    ax[r, 0].set_ylabel(path.replace("root", "R"), fontsize=8)
for q in ax.flat: q.set_xticks([]); q.set_yticks([])
fig.tight_layout(); fig.savefig(f"{a.out}/saliency_grid.png", dpi=150); plt.close(fig)

# per-depth statistics over many images: localisation + class specialisation
xs, ys = data.xte[:a.stat_images], data.yte[:a.stat_images]
by_depth, rows = {}, []
for n in model.nodes(active_only=True):
    sal = node_saliency(model, xs, n.path, n_samples=1)
    c = concentration(sal).mean().item()
    M = class_response(model, xs, ys, n.path, data.num_classes)
    spread = (M.max(0)[0] - M.min(0)[0]).max().item()     # largest between-class difference in mean gate
    by_depth.setdefault(n.depth, []).append(c)
    rows.append((n.path, n.depth, c, spread))
print("\nnode               depth  saliency-concentration(top10%)  max class-gap of mean gate")
for path, d, c, s in rows:
    print(f"{path:<18}{d:>5}  {c:>28.3f}  {s:>26.3f}")
print("\nmean concentration per depth (uniform map = 0.10):")
for d in sorted(by_depth):
    print(f"  depth {d}: {np.mean(by_depth[d]):.3f} (+-{np.std(by_depth[d]):.3f}, {len(by_depth[d])} nodes)")

fig, ax = plt.subplots(figsize=(4, 3))
ax.bar(sorted(by_depth), [np.mean(by_depth[d]) for d in sorted(by_depth)],
       yerr=[np.std(by_depth[d]) for d in sorted(by_depth)], capsize=3)
ax.axhline(0.1, c="gray", ls=":"); ax.set(xlabel="node depth", ylabel="saliency concentration")
fig.tight_layout(); fig.savefig(f"{a.out}/concentration_by_depth.png", dpi=150); plt.close(fig)

# class-response heat-maps of the selected nodes
fig, ax = plt.subplots(1, len(a.nodes), figsize=(3.2 * len(a.nodes), 3))
ax = np.atleast_1d(ax)
for q, path in zip(ax, a.nodes):
    M = class_response(model, data.xte, data.yte, path, data.num_classes)
    q.imshow(M, vmin=0, vmax=1, cmap="viridis"); q.set_title(path.replace("root", "R"), fontsize=8)
    q.set(xlabel="output k", ylabel="true class")
fig.tight_layout(); fig.savefig(f"{a.out}/class_response.png", dpi=150); plt.close(fig)
print(f"figures saved in {a.out}/")
