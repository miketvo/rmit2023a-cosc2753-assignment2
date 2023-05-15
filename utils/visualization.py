import math
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from keras.preprocessing.image import DataFrameIterator
from matplotlib import pyplot as plt
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
    ax[1].axhline(y=0.125, c='g', alpha=0.5)  # Random probability - naive classifier
    ax[1].legend(['train', 'val', 'random baseline'])

    fig.tight_layout()
    plt.show()
    if to_file is not None:
        fig.savefig(to_file)


def visualize_16predictions(
        model: Model,
        test_generator: DataFrameIterator,
        to_file: str = None
) -> None:
    fig, ax = plt.subplots(4, 4, figsize=(10, 10))
    ax = ax.ravel()

    class_labels = list(test_generator.class_indices.keys())
    for i in range(16):
        test_x, test_y = next(test_generator)
        pred = model.predict(test_x, verbose=0)
        test_x = np.squeeze(test_x)
        test_x = test_x.astype(int)

        predicted_index = np.argmax(pred)
        predicted_label = class_labels[predicted_index]
        true_index = np.argmax(test_y)
        true_label = class_labels[true_index]

        ax[i].imshow(test_x)
        ax[i].axis('off')
        ax[i].set_title(
            f'Ground Truth: {true_label}\n'
            f'Prediction: {predicted_label}',
            fontsize=8
        )
        ax[i].text(
            0.5, 1.15, f'{"CORRECT" if predicted_index == true_index else "INCORRECT"}',
            horizontalalignment='center',
            verticalalignment='bottom',
            transform=ax[i].transAxes,
            fontsize=8,
            color='green' if predicted_index == true_index else 'red',
            weight='bold'
        )

    plt.tight_layout()
    plt.show()
    if to_file is not None:
        fig.savefig(to_file)
