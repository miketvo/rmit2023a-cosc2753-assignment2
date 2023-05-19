# COSC2753 - Machine Learning: Group Machine Learning Project

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

1. `data/`: This folder contains our datasets, both raw and processed.
2. `log/`: This folder contains training logs exported from training our models.
3. `models/`: This folder contains trained models exported after training.

---


## Getting Started 🚀

Clone this repository:

```bash
git clone https://github.com/miketvo/rmit2023a-cosc2753-assignment2.git
```


### Development Environment

To set up the necessary packages for this project, run:

```bash
pip install -r requirements.txt
```

Refer to [requirements.txt](requirements.txt) for package dependencies and their versions.

<span style="color:gold">**NOTE:**</span> It is recommended that you use a Python virtual environment to avoid conflict with your global packages, and to keep your global Python installation clean. This is because we require specific versions of Numpy, Tensorflow and Keras in our code to maintain backward compatibility and compatibility between trained models and client code.

<span style="color:red">**IMPORTANT:** Required Python version: 3.10 and above, recommended Python version: 3.11.</span>


### Download Dataset

The dataset for this project is available on [Kaggle](https://kaggle.com/datasets/979207e9d5e6d91d26e8eb340941ae176c82fbdb2a25b4a436c273895ab96bb1). Follow these steps to download and set it up for training and testing:

1. Navigate to project's root directory.

2. Clean all existing files and folders in the `data/` folders (if exists) before downloading or updating this dataset:

    ```bash
    rm -r ./data/*
    ```

3. Download and **extract contents of** the `.zip` from [Kaggle](https://kaggle.com/datasets/979207e9d5e6d91d26e8eb340941ae176c82fbdb2a25b4a436c273895ab96bb1) into `data/raw` folder.

   Alternatively, use the [Kaggle CLI](https://github.com/Kaggle/kaggle-api):

    ```bash
    kaggle datasets download -d miketvo/rmit-flowers -p ./data/raw/ --unzip
    ```
   
    The resulting folder structure should look like this:
    
    ```
    .
    ├── data/
    │   └── raw/
    │       ├── Baby/
    │       ├── Calimerio/
    │       ├── Chrysanthemum/
    │       ...
    │       └── Tana/
    │
    ...
    ```
    
    where each folder corresponds to a flower class, and contains images of only that class.

    <span style="color:red">**IMPORTANT:** The location of the downloaded dataset from Kaggle MUST be in this structure, otherwise the Jupyter Notebooks and client script will not work.</span>

4. Setup for training and testing: Run [notebooks/Step2.DataPrep.ipynb](notebooks/Step2.DataPrep.ipynb) and [Step5.Recommender.ipynb](notebooks/Step5.Recommender.ipynb). They will clean, process, and split the raw dataset and the resulting train and test set into `data/train/` and `data/test/`, respectively. They will also generate a database for our image recommendation system in `data/recommender-database/`, along with `data/recommender-database.csv` that contains the feature vectors for all images in the recommender database, in addition to exporting two helper models `models/fe-cnn` and `models/clu-kmeans.model` for the recommendation system. **Note:** Clean these folders and files before you run these two notebook:

    ```bash
    rmdir -r ./data/train
    rmdir -r ./data/test
    rmdir -r ./data/recommender-database
    rm ./data/recommender-database.csv
    ```
   
    **<span style="color:red">Important:</span>** Clean and rerun this step every time you modify the raw dataset to get the most updated train dataset, test dataset, and recommender database.


### Training

Skip this step if you just want to use on of the pre-trained model packages available from [Releases](https://github.com/miketvo/rmit2023a-cosc2753-assignment2/releases).

- Run each Jupyter Notebook in `notebooks/` in their prefixed order starting `Step1.`, `Step2.`, `Step3.`, and so on, <span style="color:red">**one file at a time**</span>.
- Skip [Step2.DataPrep.ipynb](notebooks/Step2.DataPrep.ipynb) if you have already run it after downloading the raw dataset in the step above.
- Skip [Step5.Recommender.ipynb](notebooks/Step5.Recommender.ipynb) if you have already run it after downloading the raw dataset in the step above.
- The resulting models are exported into `models/` folder. Their training logs are stored in `log/` folder.

**Note:** Beware: any existing model with conflicting name in `models/` will be replaced with newly trained models.


### Using Trained Models

If you are using one of our pre-trained model packages, download your desired version from [Releases](https://github.com/miketvo/rmit2023a-cosc2753-assignment2/releases) (.zip archives) and extract its contents into this project's root directory using your preferred zip program. Make sure to check and clean `models/` folder (if exists) to avoid naming conflict with existing trained model before the extraction.

<span style="color:red">**IMPORTANT:** The location of our downloaded trained models MUST be in this folder structure, otherwise the Jupyter Notebooks and client script will not work.</span>

These trained models can then be loaded into your code with:

```python
import tensorflow as tf

model = tf.keras.models.load_model('path/to/model')
```

Additionally, two Python files, `classify.py` and `recommend.py`, are provided as simple front-ends to our trained model. You can either run them as standalone script in the terminal or import them as Python module in your own Python script or Jupyter Notebook to programmatically classify multiple images and recommend similar images for each of them.

To use them as standalone script, see instruction below:

On your terminal, make sure that you have the environment activated for the client script to have access to all required packages:

- Python Virtualenv:

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
usage: classify.py [-h] -f FILE [-c CLASSIFIER] [-g] [-v {0,1,2}]

options:
  -h, --help                                show this help message and exit
  -f FILE, --file FILE                      the image to be classified
  -c CLASSIFIER, --classifier CLASSIFIER    the machine learning model used for classification, defaults: models/clf-cnn
  -g, --gui                                 show classification result using GUI
  -v {0,1,2}, --verbose-level {0,1,2}       verbose level, default: 0
```

Example use:

```text
$ python ./classify.py -f path/to/your/your/image.png -m ./models/clf -v=1
Image image.png is classified as "Chrysanthemum" (model: "clf")
```

It also has a rudimentary GUI mode using Matplotlib, which will display the image with a caption of what flower type it is classified as:

```bash
python ./classify.py --gui -f path/to/your/your/image.png -m ./models/clf
```

**Note:** Alternatively, you can import its `classify.classify()` function into your own script or notebook to programmatically classify multiple images (see its docstring for instruction on how to use).


#### Recommending Flower Images

Use the `recommend.py` client script. Its syntax is as follows:

```text
usage: recommend.py [-h] -f FILE [-d DATABASE] [-c CLASSIFIER] [-e FEATURE_EXTRACTOR] [-k CLUSTERING_MODEL] [-n NUM]

options:
  -h, --help                                                        show this help message and exit
  -f FILE, --file FILE                                              reference image
  -d DATABASE, --database DATABASE                                  the database containing the images to be recommended, default: data/recommender-database
  -c CLASSIFIER, --classifier CLASSIFIER                            the machine learning model used for image classification, default: models/clf-cnn
  -e FEATURE_EXTRACTOR, --feature-extractor FEATURE_EXTRACTOR       the machine learning model used for image feature extraction, default: models/fe-cnn
  -k CLUSTERING_MODEL, --clustering-model CLUSTERING_MODEL          the machine learning model used for image clustering, default: models/clu-kmeans.model
  -n NUM, --num NUM                                                 number of recommendations, default: 10
```

Example:

```bash
python ./recommend.py -f path/to/your/your/image.png
```

When executed, the code above will display 10 similar flower images (GUI mode) of the same type, taken from the recommender database in `data/recommender-database/`, based on your reference image, using the default classifier, feature extractor, and clustering model
