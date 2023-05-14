import os
import pandas as pd


def load(from_dir: str) -> pd.DataFrame:
    """
    Load a directory of flower images into a Pandas ``DataFrame``. Assumes that directory contains exactly 8 folder for
    8 flower classes:

        - Baby
        - Calimerio
        - Chrysanthemum
        - Hydrangeas
        - Lisianthus
        - Pingpong
        - Rosy
        - Tana

    :param from_dir: Directory of the dataset to be loaded.
    :return: A Pandas ``DataFrame`` containing the following columns:
        - ImgPath: Path to the image.
        - Class: Type of flower in the image. Can be either one of the above eight.
    """
    if from_dir[-1] != '/' and from_dir[-1] != '\\':
        from_dir += '/'

    data_dict = {'ImgPath': [], 'Class': []}
    for flower_class in os.listdir(from_dir):
        for ImgPath in os.listdir(f'{from_dir}{flower_class}'):
            data_dict['ImgPath'].append(f'{flower_class}/{ImgPath}')
            data_dict['Class'].append(flower_class)

    return pd.DataFrame(data_dict)
