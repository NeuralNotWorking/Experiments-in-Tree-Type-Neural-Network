import torch


def node_saliency(model, x, path, target=None, n_samples=1, noise=0.1):
    # Calculate a gradient based saliency map for a tree node.
    model.eval()

    sal = torch.zeros(x.shape[0], x.shape[2], x.shape[3], device=x.device)
    sigma = noise * (x.max() - x.min()).item()

    for _ in range(n_samples):
        xi = x.clone()

        if n_samples > 1:
            xi = xi + sigma * torch.randn_like(xi)

        xi.requires_grad_(True)

        _, nodes = model(xi, return_nodes=True)
        out = nodes[path]

        if target is None:
            target_class = out.argmax(dim=1)
        else:
            target_class = torch.full(
                (x.shape[0],), target, dtype=torch.long, device=x.device
            )

        score = out.gather(1, target_class[:, None]).sum()
        grad, = torch.autograd.grad(score, xi)

        # largest gradient over the three colour channels
        sal += grad.abs().amax(dim=1)

    sal /= n_samples

    flat = sal.flatten(1)
    mn = flat.min(dim=1, keepdim=True)[0]
    mx = flat.max(dim=1, keepdim=True)[0]

    flat = (flat - mn) / (mx - mn + 1e-12)
    return flat.view_as(sal).detach()


def concentration(sal, top_frac=0.1):
    flat = sal.flatten(1)
    k = max(1, int(top_frac * flat.shape[1]))

    top = flat.topk(k, dim=1)[0].sum(dim=1)
    total = flat.sum(dim=1) + 1e-12

    return top / total


@torch.no_grad()
def class_response(model, x, y, path, num_classes):
    model.eval()

    _, nodes = model(x, return_nodes=True)
    gate = torch.sigmoid(nodes[path])

    rows = []
    for c in range(num_classes):
        mask = y == c
        if mask.any():
            rows.append(gate[mask].mean(dim=0))
        else:
            rows.append(torch.zeros(num_classes, device=x.device))

    return torch.stack(rows).cpu()
