import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from utils.glob import COLORS


def plot_class_distribution(df: pd.DataFrame, title: str, to_file: str = None):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    sns.histplot(df, x='Class', ax=ax)
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_title(title)

    for i, p in enumerate(ax.patches):
        ax.annotate(
            str(p.get_height()),
            (p.get_x() + p.get_width() / 2.,
             p.get_height()),
            ha='center', va='center', xytext=(0, 5), textcoords='offset points'
        )
        p.set_facecolor(COLORS[i])

    plt.show()
    if to_file is not None:
        fig.savefig(to_file)


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
