import os
import pandas as pd
from PIL import Image


def load(from_dir: str) -> pd.DataFrame:
    """
    Load a directory of flower images into a Pandas ``DataFrame``. Assumes that directory contains exactly 8 folders for
    8 flower classes:

        - Baby
        - Calimerio
        - Chrysanthemum
        - Hydrangeas
        - Lisianthus
        - Pingpong
        - Rosy
        - Tana

    Columns
    -------

    The resulting DataFrame has the following columns:

        - ImgPath: Path to the image.
        - FileType: The extension of the image file.
        - Width: The width (in pixels) of the image.
        - Height: The height (in pixels) of the image.
        - Ratio: The aspect ratio of the image. Calculated by taking its Width divided by its Height.
        - Mode: The mode of the image. Possible values: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#concept-modes
        - Bands: A string containing all bands of this image, separated by a space character. Read more about bands: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#bands
        - Transparency: Whether this image has transparency.
        - Animated: Whether this image has more than one frame.
        - Class: Type of flower in the image. Can be either one of the above eight.

    :param from_dir: Directory of the dataset to be loaded.
    :return: A Pandas ``DataFrame``.
    """

    if from_dir[-1] != '/' and from_dir[-1] != '\\':
        from_dir += '/'

    data_dict = {
        'ImgPath': [],
        'FileType': [],
        'Width': [],
        'Height': [],
        'Ratio': [],
        'Mode': [],
        'Bands': [],
        'Transparency': [],
        'Animated': [],
        'Class': []
    }

    for flower_class in os.listdir(from_dir):
        for ImgPath in os.listdir(f'{from_dir}{flower_class}'):
            with Image.open(f'{from_dir}{flower_class}/{ImgPath}') as im:
                data_dict['ImgPath'].append(f'{flower_class}/{ImgPath}')
                data_dict['FileType'].append(ImgPath.split('.')[-1])
                data_dict['Width'].append(im.size[0])
                data_dict['Height'].append(im.size[1])
                data_dict['Ratio'].append(im.size[0] / im.size[1])
                data_dict['Mode'].append(im.mode)
                data_dict['Bands'].append(' '.join(im.getbands()))
                data_dict['Transparency'].append(
                    True
                    if (im.mode in ('RGBA', 'RGBa', 'LA', 'La', 'PA')) or (im.mode == 'P' and 'transparency' in im.info)
                    else False
                )
                data_dict['Animated'].append(im.is_animated if hasattr(im, 'is_animated') else False)
                data_dict['Class'].append(flower_class)

    return pd.DataFrame(data_dict)
