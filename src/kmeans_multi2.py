import csv
import argparse
import numpy as np
import os
from textblob import TextBlob
from sklearn import cluster
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


def convert_to_stars(labels, sorted_centroids, mode):
    stars = []
    if mode == 0:  # means positive
        offset = [4, 5, 5, 5, 5, 5]
    else:
        offset = [1, 1, 1, 1, 1, 2]
    for l in labels:
        stars.append(offset[sorted_centroids.index(l)])
    return stars


def kmpp(data_matrix, k, mode):
    kmeans = cluster.KMeans(init='k-means++', n_clusters=k, n_init=10)
    c = kmeans.fit(data_matrix)
    centroids = c.cluster_centers_
    labels = c.labels_

    # find sorted labels
    centroids_l = centroids.tolist()
    sorted_centroids = [centroids_l.index(x) for x in sorted(centroids_l)]

    labels = convert_to_stars(labels, sorted_centroids, mode)
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



def kmeans_clustering(ip_csv, mode):
    op_csv = ip_csv + "_"
    data = []
    with open(ip_csv, "rb") as source:
        rdr = csv.reader(source)
        next(rdr)
        for r in rdr:
            data.append(float(r[3]))
    data = np.array(data).reshape(-1, 1)
    # print data
    centroids, labels = kmpp(data, 6, mode)
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
    pos_op_csv_lst = []
    neg_op_csv_lst = []
    wrts_p = []
    wrts_n = []
    i = 0
    with open(ip_csv, "rb") as source:
        rdr = csv.reader(source)
        header = next(rdr)
        while i < len(range):
            pos_op_csv_lst.append(strings[0] + "_mclusp" + str(i) + ".csv")
            neg_op_csv_lst.append(strings[0] + "_mclusn" + str(i) + ".csv")
            wrts_p.append(csv.writer(open(pos_op_csv_lst[i], "wb")))
            wrts_p[i].writerow(header)
            wrts_n.append(csv.writer(open(neg_op_csv_lst[i], "wb")))
            wrts_n[i].writerow(header)
            i += 1
        for r in rdr:
            l = len(r[1])
            # find the correct split file
            i = 0
            while l > range[i]:
                i += 1
            if (float(r[3]) >= 0):  # Polairty
                wrts_p[i].writerow(r)
            else:
                wrts_n[i].writerow(r)
        return pos_op_csv_lst, neg_op_csv_lst


def strip_reviews(ip_csv):
    op_csv = '{0}_strip.csv'.format(ip_csv.split('.csv')[0])
    strip_csv = '{0}_strip_3star.csv'.format(ip_csv.split('.csv')[0])

    with open(ip_csv, "rb") as source, open(op_csv, "wb") as result, open(strip_csv, "wb") as strip:
        rdr = csv.reader(source)
        wtr_result = csv.writer(result)
        wtr_strip = csv.writer(strip)
        header = next(rdr)
        wtr_result.writerow(header)
        wtr_strip.writerow(header)
        for r in rdr:
            if (float(r[2]) == 3):  # or (float(r[3]) == 0):
                wtr_strip.writerow(r)
            else:
                wtr_result.writerow(r)

    return op_csv


def print_analysis(pos_op_csv_lst, neg_op_csv_lst):
    files = pos_op_csv_lst[:-1] + neg_op_csv_lst[:-1]
    y_true = []
    y_pred = []
    for file in files:
        file_csv = pd.read_csv(file)
        for i, row in enumerate(file_csv.values):
            y_true.append(row[2])
            y_pred.append(row[4])

    print confusion_matrix(y_true, y_pred)
    print precision_recall_fscore_support(y_true, y_pred, average='micro')


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

    print "Extract Level 3 reviews"
    ip_csv = strip_reviews(ip_csv)

    print "Splitting files polarity"
    range = [100, 10000]
    pos_op_csv_lst, neg_op_csv_lst = split_files(ip_csv, range)

    # Positive list
    for file in pos_op_csv_lst:
        print "Staring with kmeans++ for ", file
        kmeans_clustering(file, 0)  # 0 - positive
        print "-" * 100
        print "Results stats for ", file
        print "-" * 100

    for file in neg_op_csv_lst:
        print "Staring with kmeans++ for ", file
        kmeans_clustering(file, 1)  # 1 - negative
        print "-" * 100
        print "Results stats for ", file
        print "-" * 100

    print_analysis(pos_op_csv_lst, neg_op_csv_lst)
