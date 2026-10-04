"""Experiment 1: train a tree until (near) 100% training accuracy. Saves curves + summary + checkpoint."""
import argparse, json, os
from dataclasses import asdict
from tree_nn.cli import add_data_args, add_model_args, add_train_args, make_data, make_cfg
from tree_nn.models import TreeNetwork, save_tree
from tree_nn.train import fit, set_seed, get_device, evaluate
from tree_nn.viz import plot_curves

p = argparse.ArgumentParser()
add_data_args(p); add_model_args(p); add_train_args(p)
p.add_argument("--out", default="results/overfit")
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)

set_seed(args.seed); dev = get_device()
data = make_data(args, dev)
model = TreeNetwork(make_cfg(args, data)).to(dev)
print(model.summary())
hist = fit(model, data, args.epochs, args.lr, args.wd, args.batch_size, args.stop_acc)

last = hist[-1]
summary = {**model.summary(), **{k: last[k] for k in ["train_acc", "test_acc", "train_loss", "test_loss"]},
           "epochs_run": last["epoch"], "dataset": args.dataset, "label_noise": args.label_noise,
           "train_size": len(data.xtr)}
json.dump(hist, open(f"{args.out}/history.json", "w"), indent=1)
json.dump(summary, open(f"{args.out}/summary.json", "w"), indent=1)
save_tree(model, f"{args.out}/model.pt", args)
plot_curves({"tree": hist}, f"{args.out}/curves.png", f"Tree net on {args.dataset}")
print(json.dumps(summary, indent=1))
