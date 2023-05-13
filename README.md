# Project: Classify Images of Flowers

Assume you are a team of machine learning engineers working for an ecommerce flower shop, where users can order flowers. Before users buy flowers, the systems should have a functionality to help users navigate to the type of flowers that users want to buy. In most of the current online flower shops, users should type the name of the flowers and browse from the list of the results. However, to enhance the quality of the searching results, our shop provides an image based searching function, where the users can upload the images of the flowers that they are looking for. The system will accomplish an image search and return the list of flowers which are similar to the input image from users. In the dataset Flowers, there are 08 types of flowers: Baby, Calimero, Chrysanthemum, Hydrangeas, Lisianthus, Pingpong, Rosy and Tana.


## Goals

- Goal 1: Classify images according to flower type (Baby, Calimero, Chrysanthemum, Hydrangeas, Lisianthus, Pingpong, Rosy and Tana)
- Goal 2: Recommend 10 flower images in the dataset which is similar to the input flower image from users.


## Dataset

Available on [Kaggle](https://kaggle.com/datasets/979207e9d5e6d91d26e8eb340941ae176c82fbdb2a25b4a436c273895ab96bb1). Download and **extract contents of** the `archive.zip` into `data/` folder.

Alternatively, use the [Kaggle CLI](https://github.com/Kaggle/kaggle-api):

```bash
kaggle datasets download -d miketvo/rmit-flowers -p ./data/ --unzip
```


## Requirements

- You are required to do the pre-processing step on the Flower dataset, including extracollection if necessary.
- You must investigate at least one machine learning algorithms for each of the two tasks. That is, you must build at least one model capable of predicting the type of flower images, and at least one model capable of showing 10 similar images.
- You must submit two models (one for each task).
- You are not required to use separate type(s) of machine learning algorithms, however, a thorough investigation should consider different types of algorithms.
- You are required to fully train your own algorithms. You may not use pre-trained systems which are trained on other datasets (not given to you as part of this assignment).
- For higher grades (HD/DI) you must explore how the current status of the data will affect to the result of the models, how we can improve the models, and implement your suggestion to improve the models.
- Your final report must conduct an analysis and comparison between different model results, not only just one model.


## Independent Evaluation

- Your independent evaluation is to research other works that have the same goals. Then you must compare and contrast your results to those other works.
- Using data collected completely outside of the scope of your original training and evaluation.
