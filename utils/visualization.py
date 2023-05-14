import math
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from keras.preprocessing.image import DataFrameIterator
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from keras.engine.training import Model


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


def data_plot_16samples(from_dir: str, df: pd.DataFrame, to_file: str = None):
    if from_dir[-1] != '/' and from_dir[-1] != '\\':
        from_dir += '/'

    sample_indices = np.linspace(0, df.shape[0] - 1, 16)

    fig, ax = plt.subplots(4, 4, figsize=(12, 12))
    ax = ax.ravel()

    ax_i = 0
    for i in sample_indices:
        with Image.open(f'{from_dir}{df.iloc[int(i)]["ImgPath"]}') as im:
            ax[ax_i].imshow(im)
            ax[ax_i].set_title(df.iloc[int(i)]["ImgPath"], fontsize=12)
            ax[ax_i].axis('off')

        ax_i += 1

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
    ax[1].set_xlabel('epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].axhline(y=0.125, c='gold', a=0.5)  # Random probability - naive classifier
    ax[1].legend(['train', 'val', 'random'])

    plt.show()
    if to_file is not None:
        fig.savefig(to_file)


def visualize_predictions(
        labels: list,
        model: Model,
        test_generator: DataFrameIterator,
        to_file: str = None
) -> None:
    label_names = {}
    for i, label in enumerate(labels):
        label[label] = i

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
