import numpy as np
from matplotlib import pyplot as plt


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
