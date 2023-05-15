import os
import argparse
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
from keras.engine.training import Model

from utils.glob import TARGET_IMG_SIZE
from utils.glob import CLASS_LABELS
import utils.data_manip as manip


def classify(image_path: str, model_path: str, verbose: bool = False) -> tuple:
    im_original = Image.open(image_path)
    im_processed = manip.remove_transparency(im_original)
    im_processed = manip.resize_crop(im_processed, TARGET_IMG_SIZE, TARGET_IMG_SIZE)
    im_processed = manip.normalize_pixels(im_processed)
    im_processed = tf.expand_dims(im_processed, axis=0)

    model: Model = tf.keras.models.load_model(model_path)
    pred = model.predict(im_processed, verbose=1 if verbose else 0)

    pred_class_idx = tf.argmax(pred, axis=1).numpy()[0]
    pred_class_label = CLASS_LABELS[pred_class_idx]

    return im_original, pred_class_label


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-f', '--file', required=True, help='the image to be classified')
    ap.add_argument('-c', '--classifier', required=True, help='the machine learning model used for classification')
    ap.add_argument('-g', '--gui', action='store_true', help='show classification result using GUI')
    ap.add_argument('-v', '--verbose-level', required=False, choices=['0', '1', '2'], default='0', help="verbose level, default: 0")
    args = vars(ap.parse_args())
    verbose_level = int(args['verbose_level'])

    img = os.path.abspath(args['file'])
    clf = os.path.abspath(args['classifier'])
    image, predicted_label = classify(img, clf, False if verbose_level < 1 else True)

    if args['gui']:
        canvas = ImageDraw.Draw(image)
        canvas.text(
            (10, 10),
            predicted_label,
            fill=(255, 0, 0),
            stroke_fill=(0, 0, 0),
            stroke_width=2,
            font=ImageFont.truetype(os.path.abspath('font/OpenSans-Regular.ttf'), size=24)
        )
        image.show()
    else:
        if verbose_level == 0:
            print(predicted_label)
        else:
            print(f'Image {os.path.basename(img)} is classified as "{predicted_label}" (model: "{os.path.basename(clf)}")')
