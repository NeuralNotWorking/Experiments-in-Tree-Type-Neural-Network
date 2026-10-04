from dataclasses import dataclass, asdict
from typing import Sequence, Union

import torch
import torch.nn as nn


@dataclass
class TreeConfig:
    num_classes: int = 10
    in_channels: int = 3
    depth: int = 2
    num_children: int = 2
    hidden_dim: Union[int, Sequence[int]] = 128
    feature_dim: int = 128
    channels: Sequence[int] = (32, 64, 128)
    trunk: str = "shared"

    def hidden_at(self, depth):
        if isinstance(self.hidden_dim, int):
            return self.hidden_dim

        return self.hidden_dim[min(depth, len(self.hidden_dim) - 1)]


class FeatureExtractor(nn.Module):
    def __init__(self, in_channels=3, channels=(32, 64, 128), feature_dim=128):
        super().__init__()

        layers = []
        c = in_channels

        for ch in channels:
            layers.append(nn.Conv2d(c, ch, 3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(ch))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(2))
            c = ch

        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(2)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c * 4, feature_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return self.fc(x)


class TreeNode(nn.Module):
    def __init__(self, cfg, depth=0, path="root"):
        super().__init__()

        self.depth = depth
        self.path = path

        if cfg.trunk == "separate":
            self.extractor = FeatureExtractor(
                cfg.in_channels, cfg.channels, cfg.feature_dim
            )
        else:
            self.extractor = None

        h = cfg.hidden_at(depth)

        self.mlp = nn.Sequential(
            nn.Linear(cfg.feature_dim, h),
            nn.ReLU(inplace=True),
            nn.Linear(h, cfg.num_classes)
        )

        if depth < cfg.depth:
            n_children = cfg.num_children
        else:
            n_children = 0

        self.children_nodes = nn.ModuleList([
            TreeNode(cfg, depth + 1, f"{path}/{i}")
            for i in range(n_children)
        ])

        self.child_weight = nn.Parameter(
            0.1 * torch.randn(n_children, cfg.num_classes)
        )
        self.register_buffer(
            "child_enabled", torch.ones(n_children)
        )

    def forward(self, inp, record=None):
        if self.extractor is not None:
            features = self.extractor(inp)
        else:
            features = inp

        logits = self.mlp(features)

        for i, child in enumerate(self.children_nodes):
            if self.child_enabled[i] == 0:
                continue

            child_out = child(inp, record)
            gate = torch.sigmoid(child_out)
            logits = logits + self.child_weight[i] * gate

        if record is not None:
            record[self.path] = logits

        return logits


def walk(node, active_only=False):
    yield node

    for i, child in enumerate(node.children_nodes):
        if active_only and node.child_enabled[i] == 0:
            continue

        yield from walk(child, active_only)


def _own_params(node):
    total = sum(p.numel() for p in node.mlp.parameters())
    total += node.child_weight.numel()

    if node.extractor is not None:
        total += sum(p.numel() for p in node.extractor.parameters())

    return total


class TreeNetwork(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        if cfg.trunk == "shared":
            self.trunk = FeatureExtractor(
                cfg.in_channels, cfg.channels, cfg.feature_dim
            )
        else:
            self.trunk = None

        self.root = TreeNode(cfg)

    def forward(self, x, return_nodes=False):
        if self.trunk is not None:
            inp = self.trunk(x)
        else:
            inp = x

        record = {} if return_nodes else None
        out = self.root(inp, record)

        if return_nodes:
            return out, record

        return out

    def nodes(self, active_only=False):
        return list(walk(self.root, active_only))

    def node_dict(self):
        return {node.path: node for node in self.nodes()}

    def summary(self):
        active = self.nodes(active_only=True)
        all_nodes = self.nodes()

        if self.trunk is not None:
            trunk_params = sum(
                p.numel() for p in self.trunk.parameters()
            )
        else:
            trunk_params = 0

        return {
            "depth": self.cfg.depth,
            "num_children": self.cfg.num_children,
            "hidden_dim": self.cfg.hidden_dim,
            "feature_dim": self.cfg.feature_dim,
            "trunk": self.cfg.trunk,
            "nodes_total": len(all_nodes),
            "nodes_active": len(active),
            "params_total": sum(p.numel() for p in self.parameters()),
            "params_active": trunk_params + sum(
                _own_params(node) for node in active
            )
        }


def load_tree(path, device="cpu"):
    checkpoint = torch.load(
        path, map_location=device, weights_only=False
    )

    cfg = TreeConfig(**checkpoint["cfg"])
    cfg.channels = tuple(cfg.channels)

    model = TreeNetwork(cfg).to(device)
    model.load_state_dict(checkpoint["state"])

    return model, checkpoint


def save_tree(model, path, args=None):
    torch.save({
        "cfg": asdict(model.cfg),
        "state": model.state_dict(),
        "args": vars(args) if args is not None else None
    }, path)
