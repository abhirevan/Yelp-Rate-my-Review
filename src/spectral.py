# -*- coding: utf-8 -*-
import csv
import argparse
import shutil
from textblob import TextBlob
from sklearn import cluster
import numpy as np
from numpy import *
import os

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

def gaussian_distance(data, sigma=1.0):
    m = shape(data)[0]
    adjacency = zeros((m, m))

    for i in range(0, m):
        for j in range(0, m):
            if i >= j: # since it's symmetric, just assign the upper half the same time we assign the lower half
                continue
            adjacency[j, i] = adjacency[i, j] = sum((data[i] - data[j])**2)

    adjacency = np.exp(-adjacency / (2 * sigma ** 2)) - identity(m)

    return adjacency


def spectral_clustering(ip_csv):
    op_csv = ip_csv + "_"
    data = []
    print ip_csv
    with open(ip_csv, "rb") as source:
        rdr = csv.reader(source)
        next(rdr)
        for r in rdr:
            print float(r[3])
            data.append(float(r[3]))


    #Spectral clustering with gaussian distance affintty matrix
    print "Running Spectral clustering with gaussian distance affinity matrix"
    clustering = cluster.SpectralClustering(5, affinity='precomputed', eigen_solver='arpack')
    affinity = gaussian_distance(data)
    print affinity
    clustering.fit(affinity)
    clusters = clustering.fit_predict(affinity)
    print clusters
    write_data(ip_csv,op_csv,clusters,"Spectral_Gauss")
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
    op_csv = '{0}_spec.csv'.format(ip_csv.split('.csv')[0])

    shutil.copyfile(ip_csv,op_csv)
    print "Staring with spectral_clustering++"
    spectral_clustering(op_csv)
    print "-" * 100
    #print "Predicting data"
    #print_stats(op_csv, 4)
    #print "-" * 100

