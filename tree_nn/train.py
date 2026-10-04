import random
import time

import numpy as np
import torch
import torch.nn.functional as F


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


@torch.no_grad()
def evaluate(model, x, y, bs=1000):
    model.eval()

    total_loss = 0.0
    correct = 0

    for i in range(0, len(x), bs):
        out = model(x[i:i + bs])
        target = y[i:i + bs]

        total_loss += F.cross_entropy(
            out, target, reduction="sum"
        ).item()
        correct += (out.argmax(1) == target).sum().item()

    return total_loss / len(x), correct / len(x)


def fit(model, data, epochs=100, lr=1e-3, weight_decay=0.0,
        batch_size=128, stop_train_acc=None, log_every=1, tag=""):

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    n = len(data.xtr)
    device = data.xtr.device
    history = []
    start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()

        order = torch.randperm(n, device=device)

        for i in range(0, n, batch_size):
            idx = order[i:i + batch_size]

            # BatchNorm does not work with a batch of one.
            if len(idx) < 2:
                continue

            out = model(data.xtr[idx])
            loss = F.cross_entropy(out, data.ytr[idx])

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        train_loss, train_acc = evaluate(
            model, data.xtr, data.ytr
        )
        test_loss, test_acc = evaluate(
            model, data.xte, data.yte
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc
        })

        if log_every and (epoch % log_every == 0 or epoch == 1):
            elapsed = time.time() - start
            print(
                f"{tag}ep {epoch:3d} | "
                f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
                f"test loss {test_loss:.4f} acc {test_acc:.4f} | "
                f"{elapsed:.0f}s",
                flush=True
            )

        if stop_train_acc is not None and train_acc >= stop_train_acc:
            break

    return history
