import os
import pandas as pd


def load() -> pd.DataFrame:
    data_dict = {'ImgPath': [], 'Class': []}
    for flower_class in os.listdir('../data'):
        for ImgPath in os.listdir(f'../data/{flower_class}'):
            data_dict['ImgPath'].append(f'{flower_class}/{ImgPath}')
            data_dict['Class'].append(flower_class)

    return pd.DataFrame(data_dict)
