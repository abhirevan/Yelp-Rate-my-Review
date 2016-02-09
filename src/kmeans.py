import csv
import argparse
import numpy as np
import os
from pandas import DataFrame
from textblob import TextBlob
from sklearn import cluster
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support,confusion_matrix
import shutil



def convert_to_stars(labels, sorted_centroids):
    stars = []
    for l in labels:
        stars.append(sorted_centroids.index(l) + 1)
    return stars


def kmpp(data_matrix, k):
    kmeans = cluster.KMeans(init='k-means++', n_clusters=k, n_init=10)
    c = kmeans.fit(data_matrix)
    centroids = c.cluster_centers_
    labels = c.labels_

    # find sorted labels
    centroids_l = centroids.tolist()
    sorted_centroids = [centroids_l.index(x) for x in sorted(centroids_l)]

    labels = convert_to_stars(labels, sorted_centroids)
    return centroids, labels


def write_data(ip_csv, op_csv, labels, type):
    with open(ip_csv, "rb") as source, open(op_csv, "wb") as result:
        rdr = csv.reader(source)
        wtr = csv.writer(result)
        wtr.writerow(next(rdr) + [type])
        i = 0
        for r in rdr:
            wtr.writerow((r) + [labels[i]])
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
    with open(ip_csv, "rU") as source, open(op_csv, "wb") as result:
        rdr = csv.reader(source)
        wtr = csv.writer(result)
        wtr.writerow(next(rdr) + ["Polarity"])
        i = 0
        for r in rdr:
            # print i
            blob = TextBlob(r[1].decode("utf8"))
            wtr.writerow((r) + [blob.sentiment.polarity])
            # i += 1


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


def split_files(ip_csv, range):
    strings = ip_csv.split('.csv')
    op_csv_list = []
    wrts = []
    i = 0
    with open(ip_csv, "rb") as source:
        rdr = csv.reader(source)
        header = next(rdr)
        while i < len(range):
            op_csv_list.append(strings[0] + "_clus" + str(i) + ".csv")
            wrts.append(csv.writer(open(op_csv_list[i], "wb")))
            wrts[i].writerow(header)
            i += 1

        for r in rdr:
            l = len(r[1])
            # find the correct split file
            i = 0
            while l > range[i]:
                i += 1
            wrts[i].writerow(r)

        return op_csv_list

def print_analysis(op_csv_list):
    y_true = []
    y_pred = []
    for file in op_csv_list:
        file_csv = pd.read_csv(file)
        for i, row in enumerate(file_csv.values):
            y_true.append(row[2])
            y_pred.append(row[4])

    print confusion_matrix(y_true, y_pred)
    print precision_recall_fscore_support(y_true, y_pred, average='micro')

if __name__ == '__main__':
    timestamp1 = time.time()
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

    shutil.copyfile(ip_csv,op_csv)
    print "Staring with kmeans++"
    kmeans_clustering(op_csv)
    print "-" * 100

    print_analysis([op_csv])