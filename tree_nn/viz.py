import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def plot_curves(histories, path, title=""):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))

    for i, (name, history) in enumerate(histories.items()):
        epochs = [r["epoch"] for r in history]
        color = f"C{i}"

        ax[0].plot(
            epochs,
            [r["train_loss"] for r in history],
            color=color,
            label=f"{name} train"
        )
        ax[0].plot(
            epochs,
            [r["test_loss"] for r in history],
            color=color,
            linestyle="--",
            label=f"{name} test"
        )

        ax[1].plot(
            epochs,
            [r["train_acc"] for r in history],
            color=color,
            label=f"{name} train"
        )
        ax[1].plot(
            epochs,
            [r["test_acc"] for r in history],
            color=color,
            linestyle="--",
            label=f"{name} test"
        )

    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("cross-entropy loss")
    ax[0].set_title("Loss")

    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("accuracy")
    ax[1].set_title("Accuracy")

    ax[0].legend(fontsize=7)
    ax[1].legend(fontsize=7)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_double_descent(agg, path, xlabel="number of parameters",
                        interp=None, title=""):
    x = [row["params"] for row in agg]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))

    for key, label in [
        ("test_err", "test error"),
        ("train_err", "train error")
    ]:
        ax[0].errorbar(
            x,
            [row[key] for row in agg],
            yerr=[row[key + "_std"] for row in agg],
            marker="o",
            capsize=2,
            label=label
        )

    ax[1].plot(
        x,
        [row["test_loss"] for row in agg],
        marker="o",
        color="C2"
    )

    for a in ax:
        a.set_xscale("log")
        a.set_xlabel(xlabel)

        if interp is not None:
            a.axvline(
                interp,
                color="gray",
                linestyle=":",
                label=(
                    "interpolation (train err<=1%)"
                    if a is ax[0] else None
                )
            )

    ax[0].set_ylabel("error")
    ax[1].set_ylabel("test loss")
    ax[0].legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
