import csv
import shutil
from sklearn import cluster
import numpy as np
from numpy import *
from sklearn.metrics import silhouette_score


def kmpp(data_matrix, k):
    kmeans = cluster.KMeans(init='k-means++', n_clusters=k, n_init=10)
    c = kmeans.fit(data_matrix)
    centroids = c.cluster_centers_
    labels = c.labels_

    # find sorted labels
    # centroids_l = centroids.tolist()
    # sorted_centroids = [centroids_l.index(x) for x in sorted(centroids_l)]

    # labels = convert_to_stars(labels, sorted_centroids)
    return centroids, labels


def kmeans_clustering(ip_csv, n_clusters):
    op_csv = ip_csv + "_"
    data = []
    with open(ip_csv, "rb") as source:
        rdr = csv.reader(source)
        next(rdr)
        for r in rdr:
            data.append(float(r[3]))
    data = np.array(data).reshape(-1, 1)
    # print data
    centroids, labels = kmpp(data, n_clusters)
    # print centroids
    i = 0
    for c in np.nditer(centroids):
        print "Centroid " + str(i), "->", c
        i += 1
    return centroids, labels, data


def find_silhouette_score(ip_csv, n_clusters):
    centroids, labels, data = kmeans_clustering(ip_csv, n_clusters)
    print n_clusters, "->", silhouette_score(data, labels)


if __name__ == '__main__':
    ip_csv = raw_input("Input File name?")
    op_csv = '{0}_silh.csv'.format(ip_csv.split('.csv')[0])

    shutil.copyfile(ip_csv, op_csv)

    i = 2

    while i < 11:
        find_silhouette_score(op_csv, i)
        i += 1
