# COSC2658 Data Structures and Algorithms - Group Project

For problem description and requirements, see [Project Statement](project-statement.md).

---


## Project Structure


```
.
├── font/
├── notebooks/
│   ├── images/
│   ├── Step1.EDA.ipynb
│   ├── Step2.DataPrep.ipynb
│   └── Step3.Classifier-BaselineModel.ipynb
├── scraping/
│   └── scrape.py
├── classify.py
├── recommend.py
├── requirements.txt
├── .gitignore
├── project-statement.md
├── README.md
└── LICENSE
```

1. `font/`: This folder contains the fonts used in our client script's GUI mode.
2. `notebooks/`: This folder contains all Jupyter Notebooks for this project and their exported plots in `notebooks/images/`.
3. `scrape/`: This folder contains a scraping script to get more images from the internet for our dataset. All downloaded images will also be in this folder.
4. `classify.py`: Client script for classifying flower images using trained models.
5. `recommend.py`: Client script for recommending flower images using trained models.
6. `requirements.txt`: Text file for `pip` installation of necessary packages for development environment.
7. `.gitignore`: This file contains ignore VCS ignore rules.
8. `README.md`: A text file containing useful reference information about this project, including how to run the algorithm.
9. `LICENSE`: MIT


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

- Run each Jupyter Notebook in `notebooks/` in their prefixed order starting `Step1.`, `Step2.`, `Step3.`, and so on, <span style="color:red">**one file at a time**</span>.
- Skip [Step2.DataPrep.ipynb](./notebooks/Step2.DataPrep.ipynb) if you have already run it after downloading the raw dataset.
- The resulting models are exported into `models/` folder. Their training logs are stored in `log/` folder.


### Using Trained Models

If you are using one of our pre-trained model packages, download your desired version from [Packages](https://github.com/miketvo?tab=packages&repo_name=rmit2023a-cosc2753-assignment2) (.zip archives) and extract its contents into this project's root directory using your preferred zip program.

On your terminal, make sure that you have the environment activated for the client script to have access to all required packages:

- Python Virtualenv

   ```bash
   ./venv/Scripts/activate
   ```

- Conda:

   ```bash
   conda activate ./envs
   ```

#### Classifying Flower Images

Use the `classify.py` client script. Its syntax is as follows:

```text
python ./classify.py [-h] -f FILE -m MODEL [-g] [-v {0,1,2}]

options:
  -h, --help                                show this help message and exit
  -f FILE, --file FILE                      the image to be classified
  -c CLASSIFIER, --classifier CLASSIFIER    the machine learning model used for classification
  -g, --gui                                 show classification result using GUI
  -v {0,1,2}, --verbose-level {0,1,2}       verbose level, default: 0
```

Example use:

```text
$ python ./classify.py -f path/to/your/your/image.png -m ./models/clf-baseline -v=1
Image image.png is classified as "Chrysanthemum" (model: "clf-baseline")
```

It also has a rudimentary GUI mode using your system's default GUI image viewer, which will display the image with a caption of what flower type it is classified as:

```bash
python ./classify.py --gui -f path/to/your/your/image.png -m ./models/clf-baseline
```


#### Recommending Flower Images

Use the `recommend.py` client script. Its syntax is as follows:

```text
python ./recommend.py [-h] -f FILE -r RECOMMENDER -c CLASSIFIER [-n NUM]

options:
  -h, --help                                    show this help message and exit
  -f FILE, --file FILE                          reference image
  -r RECOMMENDER, --recommender RECOMMENDER     the machine learning model used for recommendation
  -c CLASSIFIER, --classifier CLASSIFIER        the machine learning model used for classification
  -n NUM, --num NUM                             number of recommendations, default: 10
```

Example:

```bash
python ./recommend.py --gui -f path/to/your/your/image.png -r ./models/recommender -c ./models/clf-baseline
```

When executed, the code above will display (using your system's default GUI image viewer) 10 similar flower images of the same type, based on your reference image.
