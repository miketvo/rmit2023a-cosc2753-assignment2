import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def plot_data_distribution(
        df: pd.DataFrame,
        col: str | list,
        title: str | list = None,
        nrows: int = 1,
        ncols: int = 1,
        figsize: tuple = (5, 5),
        colors: list = None,
        annotate: bool = False,
        xticklabels_rotation: float = 0.0,
) -> tuple:
    if type(col) is str:
        if type(title) is not str:
            raise TypeError("utils.visualization.plot_data_distribution(): 'col' and 'title' type mismatch")
        if nrows != 1 or ncols != 1:
            raise ValueError(
                "utils.visualization.plot_data_distribution(): "
                "for single plot, 'nrows' and 'ncols' must both equal to 1"
            )

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        sns.histplot(df, x=col, ax=ax)
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(ax.get_xticklabels(), rotation=xticklabels_rotation)
        if title is not None:
            ax.set_title(title)

        for i, p in enumerate(ax.patches):
            if annotate:
                ax.annotate(
                    str(p.get_height()),
                    (p.get_x() + p.get_width() / 2.,
                     p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points'
                )
            if colors is not None:
                p.set_facecolor(colors[i])

        plt.show()
        return fig, ax
    else:
        if type(title) is not str:
            raise TypeError("utils.visualization.plot_data_distribution(): 'col' and 'title' type mismatch")
        if len(col) != len(title):
            raise ValueError("utils.visualization.plot_data_distribution(): 'col' and 'title' length mismatch")
        if len(colors) != len(col):
            raise ValueError("utils.visualization.plot_data_distribution(): 'colors' length mismatch")
        if nrows * ncols != len(col):
            raise ValueError(
                "utils.visualization.plot_data_distribution(): "
                "'nrows' and 'ncols' does not match number of plots"
            )

        fig, ax = plt.subplots(nrows, ncols, figsize=figsize)

        for i, col_ in enumerate(col):
            sns.histplot(df, x=col_, ax=ax[i])
            ax[i].set_xticks(ax[i].get_xticks())
            ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=xticklabels_rotation)
            if title is not None and title[i] is not None:
                ax[i].set_title(title[i])

            for j, p in enumerate(ax[i].patches):
                if annotate:
                    ax[i].annotate(
                        str(p.get_height()),
                        (
                            p.get_x() + p.get_width() / 2.,
                            p.get_height()
                        ),
                        ha='center', va='center', xytext=(0, 5), textcoords='offset points'
                    )
                if colors is not None and colors[i] is not None:
                    p.set_facecolor(colors[i][j])

        plt.show()
        return fig, ax


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
