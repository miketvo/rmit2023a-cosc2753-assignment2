import tkinter as tk
from PIL import ImageTk, Image
import tensorflow as tf
from keras.engine.training import Model
from utils.glob import TARGET_IMG_SIZE

import utils.data_manip as manip


def classify(image_path: str, model_path: str) -> tuple:
    im = Image.open(image_path)
    im = manip.remove_transparency(im)
    im = manip.resize_crop(im, TARGET_IMG_SIZE, TARGET_IMG_SIZE)
    im = manip.normalize_pixels(im)
    im = tf.expand_dims(im, axis=0)

    model: Model = tf.keras.models.load_model(model_path)
    pred = model.predict(im)

    pred_str = ''
    return im, pred_str


if __name__ == '__main__':
    print(
        classify(
            'data/raw/Baby/000627.jpeg',
            'models/clf-baseline'
        )
    )
