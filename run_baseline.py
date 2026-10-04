"""Experiment 5: tree vs. parameter-matched CNN-with-MLP-head and plain MLP."""
import argparse, json, os
from tree_nn.cli import add_data_args, add_model_args, add_train_args, make_data, make_cfg
from tree_nn.models import TreeNetwork
from tree_nn.baselines import CNNBaseline, MLPBaseline, match_params, count_params
from tree_nn.train import fit, set_seed, get_device
from tree_nn.viz import plot_curves

p = argparse.ArgumentParser()
add_data_args(p); add_model_args(p); add_train_args(p)
p.add_argument("--out", default="results/baseline")
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)
dev = get_device()
data = make_data(args, dev)
cfg = make_cfg(args, data)
K, in_dim = data.num_classes, data.xtr[0].numel()

set_seed(args.seed)
tree = TreeNetwork(cfg)
target = count_params(tree)
h_cnn = match_params(lambda h: CNNBaseline(K, 3, cfg.channels, cfg.feature_dim, h), target)
h_mlp = match_params(lambda h: MLPBaseline(in_dim, K, h), target)
models = {"tree": tree,
          "cnn+mlp-head": CNNBaseline(K, 3, cfg.channels, cfg.feature_dim, h_cnn),
          "mlp": MLPBaseline(in_dim, K, h_mlp)}

hists, rows = {}, []
for name, m in models.items():
    set_seed(args.seed)
    m = m.to(dev)
    print(f"\n== {name}: {count_params(m)} params (tree target {target})")
    hists[name] = fit(m, data, args.epochs, args.lr, args.wd, args.batch_size, args.stop_acc, tag=f"[{name}] ",
                      log_every=10)
    l = hists[name][-1]
    best_test = max(r["test_acc"] for r in hists[name])
    rows.append(dict(model=name, params=count_params(m), train_acc=l["train_acc"], test_acc=l["test_acc"],
                     best_test_acc=best_test, train_loss=l["train_loss"], test_loss=l["test_loss"],
                     epochs_run=l["epoch"]))
json.dump(rows, open(f"{args.out}/comparison.json", "w"), indent=1)
json.dump(hists, open(f"{args.out}/histories.json", "w"))
plot_curves(hists, f"{args.out}/curves.png", f"Tree vs baselines ({args.dataset})")
print("\n{:<14}{:>10}{:>11}{:>10}{:>10}".format("model", "params", "train_acc", "test_acc", "epochs"))
for r in rows:
    print("{:<14}{:>10}{:>11.4f}{:>10.4f}{:>10}".format(r["model"], r["params"], r["train_acc"], r["test_acc"], r["epochs_run"]))
