import argparse

from .data import load_data
from .models import TreeConfig


def add_data_args(parser):
    parser.add_argument(
        "--dataset",
        default="cifar10",
        choices=["cifar10", "cifar_catdog", "catsdogs", "synthetic"]
    )
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--train-subset", type=int, default=None)
    parser.add_argument("--label-noise", type=float, default=0.0)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)


def add_model_args(parser):
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--children", type=int, default=2)
    parser.add_argument("--hidden", type=int, nargs="+", default=[128])
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--channels", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--trunk", default="shared", choices=["shared", "separate"])


def add_train_args(parser):
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--stop-acc", type=float, default=None)


def make_data(args, device, seed=None):
    if seed is None:
        seed = args.seed

    return load_data(
        args.dataset,
        args.data_root,
        args.train_subset,
        args.label_noise,
        seed,
        args.image_size,
        device
    )


def make_cfg(args, data):
    if len(args.hidden) == 1:
        hidden = args.hidden[0]
    else:
        hidden = list(args.hidden)

    return TreeConfig(
        num_classes=data.num_classes,
        depth=args.depth,
        num_children=args.children,
        hidden_dim=hidden,
        feature_dim=args.feature_dim,
        channels=tuple(args.channels),
        trunk=args.trunk
    )
