# COSC2658 Data Structures and Algorithms - Group Project

For problem description and requirements, see [Project Statement](project-statement.md).

---


## Project Structure


```
.
├── notebooks/
│   ├── images/
│   ├── Step1.EDA.ipynb
│   ├── Step2.DataPrep.ipynb
│   └── Step3.Classifier-BaselineModel.ipynb
├── scraping/
│   └── scrape.py
├── requirements.txt
├── .gitignore
├── project-statement.md
├── README.md
└── LICENSE
```

1. `notebooks/`: This folder contains all Jupyter Notebooks for this project and their exported plots in `notebooks/images/`.
2. `scrape/`: This folder contains a scraping script to get more images from the internet for our dataset. All downloaded images will also be in this folder.
3. `requirements.txt`: Text file for `pip` installation of necessary packages for development environment.
4. `.gitignore`: This file contains ignore VCS ignore rules.
5. `README.md`: A text file containing useful reference information about this project, including how to run the algorithm.
6. `LICENSE`: MIT


Additionally, these folders will be created during dataset fetching and model training:

1. `data/`: This folder contains out datasets.
2. `logs/`: This folder contains training logs exported from training our models.
3. `models/`: This folder contains trained models exported after training.

---


## Getting Started 🚀


### Development Environment

To set up the necessary packages for this project, run:

```bash
pip install -r requirements.txt
```

Refer to [requirements.txt](requirements.txt) for package dependencies and their versions.


### Download Dataset

The dataset for this project is available on [Kaggle](https://kaggle.com/datasets/979207e9d5e6d91d26e8eb340941ae176c82fbdb2a25b4a436c273895ab96bb1). Follow these steps to download and set it up for training and testing:

1. Navigate to project's root directory.

2. Clean all existing files in the `data/` folders (if exists) before downloading or updating this dataset:

    ```bash
    rm -r ./data/*
    ```

3. Download and **extract contents of** the `.zip` from [Kaggle](https://kaggle.com/datasets/979207e9d5e6d91d26e8eb340941ae176c82fbdb2a25b4a436c273895ab96bb1) into `data/raw` folder.

   Alternatively, use the [Kaggle CLI](https://github.com/Kaggle/kaggle-api):

    ```bash
    kaggle datasets download -d miketvo/rmit-flowers -p ./data/raw/ --unzip
    ```

4. Setup for training and testing: Run [notebooks/Step2.DataPrep.ipynb](./notebooks/Step2.DataPrep.ipynb). This will clean, process, and split the raw dataset and the resulting train and test set into `data/train` and `data/test`, respectively.


### Training

Skip this step if you just want to use the pre-trained model packages available from [Packages](https://github.com/miketvo?tab=packages&repo_name=rmit2023a-cosc2753-assignment2).

- Run each Jupyter Notebook in `notebooks/` in their prefixed order starting `Step1.`, `Step2.`, `Step3.`, and so on.
- Skip [Step2.DataPrep.ipynb](./notebooks/Step2.DataPrep.ipynb) if you have already run it after downloading the raw dataset.
- The resulting models are exported into `models/` folder. Their training logs are stored in `log/` folder.


### Using Trained Models

If you are using one of our pre-trained model packages, download your desired version from [Packages](https://github.com/miketvo?tab=packages&repo_name=rmit2023a-cosc2753-assignment2) (.zip archives) and extract its contents into this project's root directory using your preferred zip program.

#### Classifying Flower Images

To be written.

#### Recommending Flower Images

To be written.
