import torch.nn as nn

from .models import FeatureExtractor


def count_params(model):
    return sum(p.numel() for p in model.parameters())


class CNNBaseline(nn.Module):
    # Same convolutional feature extractor as the tree, but with a normal MLP head.
    def __init__(self, num_classes, in_channels=3, channels=(32, 64, 128),
                 feature_dim=128, hidden_dim=256, head_layers=1):
        super().__init__()

        self.trunk = FeatureExtractor(in_channels, channels, feature_dim)

        layers = []
        d = feature_dim

        for _ in range(head_layers):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            d = hidden_dim

        layers.append(nn.Linear(d, num_classes))
        self.head = nn.Sequential(*layers)

    def forward(self, x):
        x = self.trunk(x)
        return self.head(x)


class MLPBaseline(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=512, num_layers=2):
        super().__init__()

        layers = [nn.Flatten()]
        d = input_dim

        for _ in range(num_layers):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            d = hidden_dim

        layers.append(nn.Linear(d, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def match_params(build_fn, target, lo=1, hi=32768):
    # Find the smallest hidden size giving at least target parameters.
    while lo < hi:
        mid = (lo + hi) // 2

        if count_params(build_fn(mid)) < target:
            lo = mid + 1
        else:
            hi = mid

    return lo
