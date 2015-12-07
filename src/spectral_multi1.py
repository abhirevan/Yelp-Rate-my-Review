# -*- coding: utf-8 -*-

from numpy import *
import csv
import argparse
import os
from sklearn import cluster
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


def write_data(ip_csv, op_csv, labels, type):
    with open(ip_csv, "rb") as source, open(op_csv, "wb") as result:
        rdr = csv.reader(source)
        wtr = csv.writer(result)
        wtr.writerow(next(rdr) + [type])
        i = 0
        for r in rdr:
            wtr.writerow((r) + [labels[i]])
            i += 1


def print_analysis(op_csv_list):
    y_true = []
    y_pred = []
    for file in op_csv_list:
        file_csv = pd.read_csv(file)
        for i, row in enumerate(file_csv.values):
            y_true.append(row[2])
            y_pred.append(row[5])

    print confusion_matrix(y_true, y_pred)
    print precision_recall_fscore_support(y_true, y_pred, average='micro')


def gaussian_distance(data, sigma=1.0):
    m = shape(data)[0]
    adjacency = zeros((m, m))

    for i in range(0, m):
        for j in range(0, m):
            if i >= j:  # since it's symmetric, just assign the upper half the same time we assign the lower half
                continue
            adjacency[j, i] = adjacency[i, j] = sum((data[i] - data[j]) ** 2)
    adjacency = exp(-adjacency / (2 * sigma ** 2)) - identity(m)

    return adjacency


def convert_to_stars(labels, sorted_centroids):
    stars = []
    for l in labels:
        stars.append(sorted_centroids.index(l) + 1)
    return stars


def update_labels(polarity, clusters, k):
    print "Updating labels"
    df = pd.DataFrame({'clusters': clusters, 'polarity': polarity})
    df_grouped = df.groupby('clusters')
    centroid = [0] * k
    for name, group in df_grouped:
        centroid[name] = median(group['polarity'])

    sorted_centroids = [centroid.index(x) for x in sorted(centroid)]
    labels = convert_to_stars(clusters, sorted_centroids)
    return labels


def spectral_clustering(ip_csv, k):
    op_csv = ip_csv + "_"
    data = []
    truth = []
    with open(ip_csv, "rb") as source:
        rdr = csv.reader(source)
        next(rdr)
        for r in rdr:
            data.append(float(r[3]))
            truth.append(int(r[2]))
    # Spectral clustering with gaussian distance affintty matrix
    print "Running Spectral clustering with gaussian distance affinity matrix"
    clustering = cluster.SpectralClustering(k, affinity='precomputed', eigen_solver='arpack')
    affinity = gaussian_distance(data)
    print "Calculated Gaussian distance"
    clustering.fit(affinity)
    print "Fit model"
    clusters = clustering.fit_predict(affinity)
    print "Found clusters"
    labels = update_labels(data, clusters, k)
    write_data(ip_csv, op_csv, labels, "Spectral_Gauss")
    os.remove(ip_csv)
    os.rename(op_csv, ip_csv)


def split_files(ip_csv, value_range):
    strings = ip_csv.split('.csv')
    op_csv_list = []
    wrts = []
    i = 0
    with open(ip_csv, "rb") as source:
        rdr = csv.reader(source)
        header = next(rdr)
        while i < len(value_range):
            op_csv_list.append(strings[0] + "_mclus" + str(i) + ".csv")
            wrts.append(csv.writer(open(op_csv_list[i], "wb")))
            wrts[i].writerow(header)
            i += 1

        for r in rdr:
            l = len(r[1])
            # find the correct split file
            i = 0
            while l > value_range[i]:
                i += 1
            wrts[i].writerow(r)

        return op_csv_list


def print_analysis(op_csv_list):
    y_true = []
    y_pred = []
    files = op_csv_list[:-1]
    for file in files:
        file_csv = pd.read_csv(file)
        for i, row in enumerate(file_csv.values):
            y_true.append(row[2])
            y_pred.append(row[5])
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

    print "Splitting files polarity"
    value_range = [200, 10000]
    op_csv_list = split_files(ip_csv, value_range)

    for file in op_csv_list[:-1]:
        print "Staring with kmeans++ for ", file
        spectral_clustering(file, 5)
        print "-" * 100
    print_analysis(op_csv_list)
