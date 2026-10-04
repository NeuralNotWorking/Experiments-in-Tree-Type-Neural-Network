"""Experiment 2: sweep model capacity, train every model for the SAME fixed number of epochs
(no early stopping), record final train/test error vs #parameters.

Tips: use --label-noise 0.1..0.2 and a --train-subset (e.g. 10000): label noise makes the
interpolation peak much more visible, and a subset keeps the sweep affordable. Report both
the noisy and the clean (0) setting. Results are resumable (re-run the same command)."""
import argparse, copy, json, os
import numpy as np
from tree_nn.cli import add_data_args, add_model_args, add_train_args, make_data, make_cfg
from tree_nn.models import TreeNetwork
from tree_nn.train import fit, set_seed, get_device
from tree_nn.viz import plot_double_descent

p = argparse.ArgumentParser()
add_data_args(p); add_model_args(p); add_train_args(p)
p.add_argument("--vary", default="hidden", choices=["hidden", "children", "depth", "trunk"])
p.add_argument("--values", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
p.add_argument("--seeds", type=int, default=2)
p.add_argument("--out", default="results/double_descent")
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)
res_path = f"{args.out}/raw_{args.vary}.json"
raw = json.load(open(res_path)) if os.path.exists(res_path) else []
done = {(r["value"], r["seed"]) for r in raw}
dev = get_device()


def variant(cfg, v):
    cfg = copy.deepcopy(cfg)
    if args.vary == "hidden": cfg.hidden_dim = v
    elif args.vary == "children": cfg.num_children = v
    elif args.vary == "depth": cfg.depth = v
    else: cfg.channels, cfg.feature_dim = (v, 2 * v, 4 * v), 4 * v    # scale the CNN trunk width
    return cfg


for seed in range(args.seeds):
    data = make_data(args, dev, seed=args.seed + seed)
    for v in args.values:
        if (v, seed) in done:
            continue
        set_seed(args.seed + seed)
        model = TreeNetwork(variant(make_cfg(args, data), v)).to(dev)
        s = model.summary()
        print(f"\n### {args.vary}={v} seed={seed} params={s['params_total']} nodes={s['nodes_total']}", flush=True)
        hist = fit(model, data, args.epochs, args.lr, args.wd, args.batch_size, None, log_every=max(1, args.epochs // 5))
        l = hist[-1]
        raw.append(dict(vary=args.vary, value=v, seed=seed, params=s["params_total"], nodes=s["nodes_total"],
                        train_err=1 - l["train_acc"], test_err=1 - l["test_acc"],
                        train_loss=l["train_loss"], test_loss=l["test_loss"],
                        curve=[(r["epoch"], 1 - r["train_acc"], 1 - r["test_acc"]) for r in hist]))
        json.dump(raw, open(res_path, "w"))

# aggregate over seeds
agg = []
for v in sorted({r["value"] for r in raw}):
    rs = [r for r in raw if r["value"] == v]
    a = dict(value=v, params=int(np.mean([r["params"] for r in rs])), nodes=rs[0]["nodes"], n_seeds=len(rs))
    for k in ["train_err", "test_err", "test_loss"]:
        a[k] = float(np.mean([r[k] for r in rs])); a[k + "_std"] = float(np.std([r[k] for r in rs]))
    agg.append(a)
agg.sort(key=lambda a: a["params"])
interp = next((a["params"] for a in agg if a["train_err"] <= 0.01), None)
plot_double_descent(agg, f"{args.out}/dd_{args.vary}.png", f"parameters (varying {args.vary})", interp,
                    f"{args.dataset}, label noise {args.label_noise}, {args.epochs} epochs")
json.dump(agg, open(f"{args.out}/agg_{args.vary}.json", "w"), indent=1)

print("\nvalue  params   train_err  test_err(+-std)")
for a in agg:
    print(f"{a['value']:5d} {a['params']:8d}   {a['train_err']:.4f}    {a['test_err']:.4f} (+-{a['test_err_std']:.4f})")
print("interpolation threshold (first size with train err <= 1%):", interp)

# crude, honest check: is there a bump in test error followed by a decrease?
te = np.array([a["test_err"] for a in agg]); sd = np.array([a["test_err_std"] for a in agg])
if len(te) >= 4:
    peak = int(np.argmax(te[1:-1])) + 1
    rise, drop = te[peak] - te[:peak].min(), te[peak] - te[peak:].min()
    noise = max(sd.max(), 0.005)
    print(f"peak at idx {peak} (params {agg[peak]['params']}): rise {rise:.4f}, later drop {drop:.4f}, seed-std {noise:.4f}")
    print("-> rise and drop both > 2x seed-std: possible double descent, verify with more seeds/epochs"
          if rise > 2 * noise and drop > 2 * noise else
          "-> no clear double descent beyond seed noise in this sweep (do NOT claim it)")
