import argparse
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
    if num_recommendations < 1:
        raise ValueError('Number of recommendations cannot be smaller than 1. Valid range: 1 - 16. Support for more recommendations will be coming soon.')
    if num_recommendations > 16:
        raise ValueError('Number of recommendations too large. Support for more recommendations will be coming soon.')

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
    sorted_ref_cluster_indices = np.argsort(-cosine_similarities.flatten())  # Updated line
    top_ref_cluster_indices = sorted_ref_cluster_indices[:num_recommendations]
    recommendations = recommendations.iloc[top_ref_cluster_indices]

    return list(recommendations['ImgPath'].values)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-f', '--file', required=True, help='reference image')
    ap.add_argument('-d', '--database', default='data/recommender-database', help='the database containing the images to be recommended')
    ap.add_argument('-c', '--classifier', default='models/clf-cnn', help='the machine learning model used for image classification')
    ap.add_argument('-e', '--feature-extractor', default='models/fe-cnn', help='the machine learning model used for image feature extraction')
    ap.add_argument('-k', '--clustering-model', default='models/clu-kmeans.model', help='the machine learning model used for image clustering')
    ap.add_argument('-n', '--num', required=False, default='10', help="number of recommendations, default: 10")
    args = vars(ap.parse_args())
    num = int(args['num'])

    rec_paths = recommend(
        args['file'], int(args['num']),
        args['database'] + '.csv', args['classifier'], args['feature_extractor'], args['clustering_model']
    )

    fig, ax = plt.subplots(1, num, figsize=(16, 16))
    for i, rec_path in enumerate(rec_paths):
        with Image.open(f'{args["database"]}/{rec_path}') as rec:
            ax[i].imshow(rec)
            ax[i].axis('off')

    plt.tight_layout()
    plt.show()
