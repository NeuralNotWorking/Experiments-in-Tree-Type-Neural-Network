from .train import evaluate


def active_edges(model):
    edges = []

    for node in model.nodes(active_only=True):
        for i in range(len(node.children_nodes)):
            if node.child_enabled[i] > 0:
                edges.append((node, i))

    return edges


def greedy_prune(model, x, y, verbose=True):
    summary = model.summary()
    acc = evaluate(model, x, y)[1]

    log = [{
        "path": None,
        "idx": None,
        "train_acc": acc,
        "nodes_active": summary["nodes_active"],
        "params_active": summary["params_active"]
    }]

    while True:
        edges = active_edges(model)

        if not edges:
            break

        best = None

        for node, i in edges:
            node.child_enabled[i] = 0.0
            acc = evaluate(model, x, y)[1]
            node.child_enabled[i] = 1.0

            if best is None or acc > best[0]:
                best = (acc, node, i)

        acc, node, i = best
        node.child_enabled[i] = 0.0

        summary = model.summary()

        log.append({
            "path": node.path,
            "idx": i,
            "train_acc": acc,
            "nodes_active": summary["nodes_active"],
            "params_active": summary["params_active"]
        })

        if verbose:
            print(
                f"pruned {node.path}[{i}] -> "
                f"nodes {summary['nodes_active']:3d} "
                f"params {summary['params_active']:8d} "
                f"train acc {acc:.4f}",
                flush=True
            )

    return log


def replay_prunes(model, log, upto):
    nodes = model.node_dict()

    for row in log[1:upto + 1]:
        nodes[row["path"]].child_enabled[row["idx"]] = 0.0
