"""Experiment 4: network minimisation.
  sweep : shrink hidden dim / depth / #children one at a time; train each model to --stop-acc.
  prune : train (or load) a big tree, greedily remove subtrees, optionally fine-tune the result."""
import argparse, copy, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tree_nn.cli import add_data_args, add_model_args, add_train_args, make_data, make_cfg
from tree_nn.models import TreeNetwork, load_tree, save_tree
from tree_nn.prune import greedy_prune, replay_prunes
from tree_nn.train import fit, set_seed, get_device, evaluate

p = argparse.ArgumentParser()
add_data_args(p); add_model_args(p); add_train_args(p)
p.add_argument("--mode", required=True, choices=["sweep", "prune"])
p.add_argument("--target-acc", type=float, default=0.99, help="'fits the training set' threshold")
p.add_argument("--hidden-values", type=int, nargs="+", default=[256, 128, 64, 32, 16, 8, 4])
p.add_argument("--depth-values", type=int, nargs="+", default=[3, 2, 1, 0])
p.add_argument("--children-values", type=int, nargs="+", default=[4, 3, 2, 1])
p.add_argument("--ckpt", default=None, help="prune mode: start from a trained model")
p.add_argument("--finetune-epochs", type=int, default=10)
p.add_argument("--out", default="results/minimize")
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)
dev = get_device()
data = make_data(args, dev)
stop = args.stop_acc or args.target_acc

if args.mode == "sweep":
    base, rows = make_cfg(args, data), []
    for vary, values in [("hidden_dim", args.hidden_values), ("depth", args.depth_values),
                         ("num_children", args.children_values)]:
        for v in values:
            cfg = copy.deepcopy(base); setattr(cfg, vary, v)
            set_seed(args.seed)
            m = TreeNetwork(cfg).to(dev); s = m.summary()
            print(f"\n### {vary}={v}: {s['params_total']} params, {s['nodes_total']} nodes", flush=True)
            h = fit(m, data, args.epochs, args.lr, args.wd, args.batch_size, stop, log_every=0)
            l = h[-1]
            rows.append(dict(vary=vary, value=v, params=s["params_total"], nodes=s["nodes_total"],
                             train_acc=l["train_acc"], test_acc=l["test_acc"], epochs=l["epoch"],
                             reached=l["train_acc"] >= stop))
            print(rows[-1], flush=True)
            json.dump(rows, open(f"{args.out}/sweep.json", "w"), indent=1)
    print(f"\n{'vary':<13}{'value':>6}{'params':>9}{'nodes':>6}{'train_acc':>10}{'test_acc':>9}{'epochs':>7}  reached")
    for r in rows:
        print(f"{r['vary']:<13}{r['value']:>6}{r['params']:>9}{r['nodes']:>6}{r['train_acc']:>10.4f}"
              f"{r['test_acc']:>9.4f}{r['epochs']:>7}  {r['reached']}")
    fig, ax = plt.subplots(figsize=(6, 4))
    for vary in ["hidden_dim", "depth", "num_children"]:
        rs = [r for r in rows if r["vary"] == vary]
        ax.plot([r["params"] for r in rs], [r["train_acc"] for r in rs], "o-", label=vary)
    ax.axhline(stop, c="gray", ls=":"); ax.set_xscale("log")
    ax.set(xlabel="parameters", ylabel="final train accuracy"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{args.out}/sweep.png", dpi=150)

else:
    if args.ckpt:
        model, ck = load_tree(args.ckpt, dev)
    else:
        set_seed(args.seed)
        model = TreeNetwork(make_cfg(args, data)).to(dev)
        fit(model, data, args.epochs, args.lr, args.wd, args.batch_size, stop, log_every=10)
        save_tree(model, f"{args.out}/big_model.pt", args)
    state0 = copy.deepcopy(model.state_dict())
    n_search = min(len(data.xtr), 10000)           # prune search on a fixed train subset for speed
    log = greedy_prune(model, data.xtr[:n_search], data.ytr[:n_search])
    # full-train-set accuracy after each step (replaying the log on the original weights)
    curve = []
    for k in range(len(log)):
        model.load_state_dict(state0); replay_prunes(model, log, k)
        s = model.summary()
        curve.append(dict(step=k, nodes=s["nodes_active"], params=s["params_active"],
                          train_acc=evaluate(model, data.xtr, data.ytr)[1],
                          test_acc=evaluate(model, data.xte, data.yte)[1]))
    ok = [c for c in curve if c["train_acc"] >= args.target_acc]
    best = min(ok, key=lambda c: c["params"]) if ok else curve[0]
    print("\nsmallest pruned tree (no fine-tuning) with train acc >=", args.target_acc, ":", best)
    model.load_state_dict(state0); replay_prunes(model, log, best["step"])
    if args.finetune_epochs > 0:
        h = fit(model, data, args.finetune_epochs, args.lr * 0.3, args.wd, args.batch_size, None, log_every=5, tag="[finetune] ")
        best["finetuned_train_acc"], best["finetuned_test_acc"] = h[-1]["train_acc"], h[-1]["test_acc"]
        print("after fine-tuning:", best)
    json.dump(dict(curve=curve, chosen=best), open(f"{args.out}/prune.json", "w"), indent=1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([c["params"] for c in curve], [c["train_acc"] for c in curve], "o-", label="train acc")
    ax.plot([c["params"] for c in curve], [c["test_acc"] for c in curve], "s--", label="test acc")
    ax.axhline(args.target_acc, c="gray", ls=":"); ax.set_xscale("log")
    ax.set(xlabel="active parameters", ylabel="accuracy (no fine-tuning)"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{args.out}/prune.png", dpi=150)
