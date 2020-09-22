from sklearn.cluster import MiniBatchKMeans
import numpy as np
import pandas as pd
import os


def clustering(dict, n_clusters, batch_size):
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size).fit(dict)
    return kmeans


def gen_histogram(feature_vectors, kmeans):
    histogram = np.zeros(len(kmeans.cluster_centers_))
    cluster_result = kmeans.predict(feature_vectors)
    for i in cluster_result:
        histogram[i] += 1.0
    return histogram


def gen_vbow(input_path, output_name, kmeans, class_id):
    vbow = []
    listing = os.listdir(input_path)
    for csv_name in listing:
        feature_vectors = pd.read_csv(input_path+csv_name)
        histogram = gen_histogram(feature_vectors, kmeans)
        histogram = np.append(histogram, class_id)
        vbow.append(histogram)
    np.savetxt(output_name+".csv", vbow, delimiter=",")


if __name__ == '__main__':
    dict = pd.read_csv("dict.csv")
    kmeans = clustering(dict, 10, 50)
    gen_vbow(r"/home/user/input_path/", "output_name", kmeans)


