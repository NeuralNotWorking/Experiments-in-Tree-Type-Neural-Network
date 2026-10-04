import glob
import os
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Data:
    xtr: torch.Tensor
    ytr: torch.Tensor
    ytr_clean: torch.Tensor
    xte: torch.Tensor
    yte: torch.Tensor
    num_classes: int
    mean: torch.Tensor
    std: torch.Tensor
    class_names: list


def _cifar(root):
    import torchvision

    train = torchvision.datasets.CIFAR10(
        root, train=True, download=True
    )
    test = torchvision.datasets.CIFAR10(
        root, train=False, download=True
    )

    return (
        train.data,
        np.array(train.targets),
        test.data,
        np.array(test.targets),
        train.classes
    )


def _catsdogs(root, size, seed):
    from PIL import Image

    cache = os.path.join(root, f"catsdogs_{size}.npz")

    if not os.path.exists(cache):
        folder = os.path.join(root, "PetImages")

        if not os.path.isdir(folder):
            raise FileNotFoundError(
                f"Put the Cats-vs-Dogs images in {folder}/Cat and {folder}/Dog"
            )

        images = []
        labels = []

        for label, name in enumerate(["Cat", "Dog"]):
            files = sorted(glob.glob(os.path.join(folder, name, "*.jpg")))

            for file in files:
                try:
                    img = Image.open(file).convert("RGB")
                    img = img.resize((size, size), Image.BILINEAR)
                except Exception:
                    # Some files in the original dataset are corrupted.
                    continue

                images.append(np.asarray(img))
                labels.append(label)

        np.savez_compressed(
            cache,
            x=np.stack(images),
            y=np.array(labels)
        )

    data = np.load(cache)
    x = data["x"]
    y = data["y"]

    # Keep the split fixed so different runs use the same train/test sets.
    rng = np.random.default_rng(1234)
    order = rng.permutation(len(x))
    x = x[order]
    y = y[order]

    split = int(0.8 * len(x))

    return (
        x[:split],
        y[:split],
        x[split:],
        y[split:],
        ["cat", "dog"]
    )


def _synthetic(seed, k=4, n_tr=600, n_te=200):
    rng = np.random.default_rng(seed)

    templates = rng.normal(size=(k, 32, 32, 3))

    def make_data(n):
        y = rng.integers(0, k, n)
        noise = rng.normal(size=(n, 32, 32, 3))
        x = 128 + 40 * (0.6 * templates[y] + noise)
        x = np.clip(x, 0, 255).astype(np.uint8)
        return x, y

    xtr, ytr = make_data(n_tr)
    xte, yte = make_data(n_te)

    return xtr, ytr, xte, yte, [str(i) for i in range(k)]


def add_label_noise(y, frac, num_classes, rng):
    y = y.copy()

    if frac > 0:
        n = int(frac * len(y))
        idx = rng.choice(len(y), n, replace=False)

        # Make sure the new class is different from the original one.
        shift = rng.integers(1, num_classes, size=n)
        y[idx] = (y[idx] + shift) % num_classes

    return y


def load_data(name="cifar10", root="data", train_subset=None,
              label_noise=0.0, seed=0, image_size=32, device="cpu"):

    if name == "cifar10":
        xtr, ytr, xte, yte, names = _cifar(root)

    elif name == "cifar_catdog":
        xtr, ytr, xte, yte, _ = _cifar(root)

        def select_catdog(x, y):
            mask = (y == 3) | (y == 5)
            return x[mask], (y[mask] == 5).astype(np.int64)

        xtr, ytr = select_catdog(xtr, ytr)
        xte, yte = select_catdog(xte, yte)
        names = ["cat", "dog"]

    elif name == "catsdogs":
        xtr, ytr, xte, yte, names = _catsdogs(root, image_size, seed)

    elif name == "synthetic":
        xtr, ytr, xte, yte, names = _synthetic(seed)

    else:
        raise ValueError(name)

    rng = np.random.default_rng(seed)

    if train_subset is not None and train_subset < len(xtr):
        idx = rng.permutation(len(xtr))[:train_subset]
        xtr = xtr[idx]
        ytr = ytr[idx]

    num_classes = len(names)
    ytr_noisy = add_label_noise(ytr, label_noise, num_classes, rng)

    def to_tensor(x):
        x = np.ascontiguousarray(x)
        x = torch.from_numpy(x).float() / 255.0
        return x.permute(0, 3, 1, 2)

    xtr = to_tensor(xtr)
    xte = to_tensor(xte)

    mean = xtr.mean((0, 2, 3), keepdim=True)
    std = xtr.std((0, 2, 3), keepdim=True)

    xtr = (xtr - mean) / std
    xte = (xte - mean) / std

    def labels(a):
        return torch.from_numpy(a).long().to(device)

    return Data(
        xtr.to(device),
        labels(ytr_noisy),
        labels(ytr),
        xte.to(device),
        labels(yte),
        num_classes,
        mean.to(device),
        std.to(device),
        names
    )
