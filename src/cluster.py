import csv
import argparse
import numpy as np
import os
from textblob import TextBlob
from sklearn import cluster
import shutil



def kmpp(data_matrix, k):
    kmeans = cluster.KMeans(init='k-means++', n_clusters=k, n_init=10)
    c = kmeans.fit(data_matrix)
    centroids = c.cluster_centers_
    labels = c.labels_
    return centroids, labels


def aggclustering(data_matrix, k, linkage, connectivity):
    model = cluster.AgglomerativeClustering(linkage=linkage,
                                            connectivity=connectivity,
                                            n_clusters=k)
    c = model.fit(data_matrix)
    labels = c.labels_
    return labels


def write_data(ip_csv, op_csv, labels, type):
    with open(ip_csv, "rb") as source, open(op_csv, "wb") as result:
        rdr = csv.reader(source)
        wtr = csv.writer(result)
        wtr.writerow(next(rdr) + [type])
        i = 0
        for r in rdr:
            wtr.writerow((r) + [labels[i] + 1])
            i += 1


def print_stats(ip_csv, col):
    stats = {}
    i = 1
    while i <= 5:
        j = 1
        while j <= 5:
            stats[str(i) + "-" + str(j)] = 0
            j += 1
        i += 1
    with open(ip_csv, "rb") as source:
        rdr = csv.reader(source)
        next(rdr)
        for r in rdr:
            t = int(r[2])  # truth
            c = int(r[col])  # result
            k = str(c) + "-" + str(t)
            stats[k] = stats[k] + 1

    i = 1
    while i <= 5:
        j = 1
        print "For cluster: ", i
        while j <= 5:
            k = str(i) + "-" + str(j)
            print k, " -> ", stats[k]
            j += 1
        i += 1


def extract_sentiment(ip_csv, op_csv):
    # op_csv = ip_csv + "_"
    with open(ip_csv, "rb") as source, open(op_csv, "wb") as result:
        rdr = csv.reader(source)
        wtr = csv.writer(result)
        wtr.writerow(next(rdr) + ["Polarity"])
        i = 0
        for r in rdr:
            blob = TextBlob(r[1].decode("utf8"))
            wtr.writerow((r) + [blob.sentiment.polarity])


def kmeans_clustering(ip_csv):
    op_csv = ip_csv + "_"
    data = []
    with open(ip_csv, "rb") as source:
        rdr = csv.reader(source)
        next(rdr)
        for r in rdr:
            data.append(float(r[3]))
    data = np.array(data).reshape(-1, 1)
    # print data
    centroids, labels = kmpp(data, 5)
    # print centroids
    i = 0
    for c in np.nditer(centroids):
        print "Centroid " + str(i), "->", c
        i += 1
    write_data(ip_csv, op_csv, labels, "Kmeans++")
    os.remove(ip_csv)
    os.rename(op_csv, ip_csv)


def agglomerative_clustering(ip_csv, linkage, connectivity):
    op_csv = ip_csv + "_"
    data = []
    with open(ip_csv, "rb") as source:
        rdr = csv.reader(source)
        next(rdr)
        for r in rdr:
            data.append(float(r[3]))
    data = np.array(data).reshape(-1, 1)
    # print data
    labels = aggclustering(data, 5, linkage, connectivity)
    # print centroids

    write_data(ip_csv, op_csv, labels, "Agglomerative_ward")
    os.remove(ip_csv)
    os.rename(op_csv, ip_csv)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Find clustering for csv',
    )
    parser.add_argument(
        'ip_csv',
        type=str,
    )
    args = parser.parse_args()
    ip_csv = args.ip_csv
    op_csv = '{0}_clus.csv'.format(ip_csv.split('.csv')[0])


    print "-" * 100
    print "Updating polarity"
    extract_sentiment(ip_csv, op_csv)
    print "-" * 100



    print "Staring with kmeans++"
    kmeans_clustering(op_csv)
    print "-" * 100
    print "Predicting data"
    print_stats(op_csv, 4)
    print "-" * 100

