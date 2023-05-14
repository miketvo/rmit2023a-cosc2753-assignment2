import math
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def data_countplot(
        df: pd.DataFrame,
        col: str,
        ax,
        horizontal: bool = False,
        title: str = None,
        annotate: bool = False,
        palette=None,
        xticklabels_rotation: float = 0.0,
) -> None:
    if horizontal:
        sns.countplot(y=df[col], ax=ax, palette=palette)
    else:
        sns.countplot(x=df[col], ax=ax, palette=palette)
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(ax.get_xticklabels(), rotation=xticklabels_rotation)

    if title is not None:
        ax.set_title(title)

    for patch in ax.patches:
        if annotate:
            if horizontal:
                ax.annotate(
                    str(math.floor(patch.get_width())),
                    (
                        patch.get_width() / 2.,
                        patch.get_y() + patch.get_height() / 2.
                    ),
                    ha='center', va='center', xytext=(0, 0), textcoords='offset points'
                )
            else:
                ax.annotate(
                    str(math.floor(patch.get_height())),
                    (
                        patch.get_x() + patch.get_width() / 2.,
                        patch.get_height()
                    ),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points'
                )


def data_histplot(
        df: pd.DataFrame,
        col: str,
        ax,
        title: str = None,
        stat: str = 'count',
        bins='auto',
        kde: bool = False,
        line_kws: dict = None,
        annotate: bool = False,
        palette=None,
        xticklabels_rotation: float = 0.0,
) -> None:
    sns.histplot(df, x=col, ax=ax, bins=bins, kde=kde, line_kws=line_kws, stat=stat, palette=palette)
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=xticklabels_rotation)
    if title is not None:
        ax.set_title(title)

    for patch in ax.patches:
        if annotate:
            ax.annotate(
                str(patch.get_height()),
                (
                    patch.get_x() + patch.get_width() / 2.,
                    patch.get_height()
                ),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points'
            )


def plot_learning_curve(
        train_loss, val_loss, train_metric, val_metric,
        to_file: str = None
) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].plot(train_loss, 'r--')
    ax[0].plot(val_loss, 'b--')
    ax[0].set_xlabel("epochs")
    ax[0].set_ylabel("Loss")
    ax[0].legend(['train', 'val'])

    ax[1].plot(train_metric, 'r--')
    ax[1].plot(val_metric, 'b--')
    ax[1].set_xlabel("epochs")
    ax[1].set_ylabel('Accuracy')
    ax[1].legend(['train', 'val'])

    plt.show()
    if to_file is not None:
        fig.savefig(to_file)


def visualize_predictions(model, test_generator, to_file: str = None) -> None:
    label_names = {
        'Babi': 0,
        'Calimerio': 1,
        'Chrysanthemum': 2,
        'Hydrangeas': 3,
        'Lisianthus': 4,
        'Pingpong': 5,
        'Rosy': 6,
        'Tana': 7
    }

    fig, axes = plt.subplots(1, 5, figsize=(16, 4))
    batches = 0
    d_inv = {v: k for k, v in label_names.items()}

    for x, y in test_generator:
        batches = batches + 1
        y_hat = model.predict(x, verbose=0)
        x = np.squeeze(x)

        if batches <= 5:
            ax = axes[batches - 1]
            ax.imshow(x)
            ax.set_title(f'GT-{d_inv[np.argmax(y[0])]}, Pred-{d_inv[np.argmax(y_hat[0])]}', fontsize=8)
            ax.axis('off')

        else:
            break

    plt.show()
    if to_file is not None:
        fig.savefig(to_file)
