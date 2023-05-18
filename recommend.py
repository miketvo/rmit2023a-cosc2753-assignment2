import argparse
import os.path
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

from classify import classify


def recommend(
        ref_path: str, num_recommendations: int,
        data_path: str, clf_path: str, fe_path: str, clu_path: str,
) -> list:
    """
    Recommends similar images based on a reference image.

    :param ref_path: Path to the reference image.
    :param num_recommendations: Number of recommended images to return.
    :param data_path: Path to the .csv data file containing recommender database image feature vectors. This file must be generated using the same feature extractor specified in fe_path.
    :param clf_path: Path to the classifier model file.
    :param fe_path: Path to the feature extraction model file.
    :param clu_path: Path to the clustering model file.
    :return: List of paths to the recommended images.
    """
    if num_recommendations < 1:
        raise ValueError('Number of recommendations cannot be smaller than 1.')

    df_rec = pd.read_csv(data_path)
    fe = tf.keras.models.load_model(fe_path)
    clu = joblib.load(clu_path)
    clu.set_params(n_clusters=int(np.sqrt(len(df_rec) / num_recommendations)))

    ref_processed, ref_class = classify(ref_path, classifier_path=clf_path, return_original=False, verbose=False)
    recommendations = df_rec[df_rec['Class'] == ref_class]

    # Extract reference image feature vector
    ref_processed = np.squeeze(ref_processed)
    ref_feature_vector = fe.predict(
        tf.expand_dims(ref_processed, axis=0),
        verbose=0
    )
    ref_feature_vector = ref_feature_vector.astype(float)
    ref_feature_vector = ref_feature_vector.reshape(1, -1)

    # Cluster reference image
    clu.fit(recommendations.drop(['ImgPath', 'Class'], axis='columns').values)
    ref_cluster = clu.predict(ref_feature_vector)
    ref_cluster_indices = np.where(clu.labels_ == ref_cluster)[0]
    recommendations = recommendations.iloc[ref_cluster_indices]

    # Rank cluster and produce top cosine similarity recommendations
    cosine_similarities = cosine_similarity(
        ref_feature_vector,
        recommendations.drop(['ImgPath', 'Class'], axis='columns')
    )
    sorted_ref_cluster_indices = np.argsort(-cosine_similarities.flatten())
    if num_recommendations > len(sorted_ref_cluster_indices):
        raise ValueError('Number of recommendations too large. Insufficient database size.')
    top_ref_cluster_indices = sorted_ref_cluster_indices[:num_recommendations]
    recommendations = recommendations.iloc[top_ref_cluster_indices]

    return list(recommendations['ImgPath'].values)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-f', '--file', required=True, help='reference image')
    ap.add_argument('-d', '--database', default='data/recommender-database', help='the database containing the images to be recommended, default: data/recommender-database')
    ap.add_argument('-c', '--classifier', default='models/clf-cnn', help='the machine learning model used for image classification, default: models/clf-cnn')
    ap.add_argument('-e', '--feature-extractor', default='models/fe-cnn', help='the machine learning model used for image feature extraction, default: models/fe-cnn')
    ap.add_argument('-k', '--clustering-model', default='models/clu-kmeans.model', help='the machine learning model used for image clustering, default: models/clu-kmeans.model')
    ap.add_argument('-n', '--num', required=False, default='10', help="number of recommendations, default: 10")
    args = vars(ap.parse_args())
    num = int(args['num'])

    fig, axes = plt.subplots(max([1, num // 5]) + 1, 5, figsize=(16, 16), num='Flower Image Recommender')
    axes = axes.ravel()

    ref = Image.open(args['file'])
    _, ref_class = classify(args['file'], classifier_path=args['classifier'], return_original=False, verbose=False)
    axes[2].imshow(ref)
    axes[2].set_title(
        f'Reference Image - "{ref_class}"',
        fontsize=10,
        weight='bold'
    )
    axes[2].text(
        0.5, -0.08, f'{os.path.relpath(args["file"])}',
        horizontalalignment='center',
        verticalalignment='center_baseline',
        transform=axes[2].transAxes,
        fontsize=8,
    )
    for i, rec_path in enumerate(recommend(
        args['file'], int(args['num']),
        args['database'] + '.csv', args['classifier'], args['feature_extractor'], args['clustering_model']
    ), start=5):
        with Image.open(f'{args["database"]}/{rec_path}') as rec:
            axes[i].imshow(rec)

    for ax in axes:
        ax.axis('off')

    plt.show()
