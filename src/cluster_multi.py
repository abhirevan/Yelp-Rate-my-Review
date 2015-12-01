import csv
import argparse
import numpy as np
import os
from textblob import TextBlob
from sklearn import cluster


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
    with open(ip_csv, "rU") as source, open(op_csv, "wb") as result:
        rdr = csv.reader(source)
        wtr = csv.writer(result)
        wtr.writerow(next(rdr) + ["Polarity"])
        i = 0
        for r in rdr:
            #print i
            blob = TextBlob(r[1].decode("utf8"))
            wtr.writerow((r) + [blob.sentiment.polarity])
            #i += 1


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


def split_files(op_csv, range):
    strings = op_csv.split('.csv')
    op_csv_list = []
    wrts = []
    i = 0
    with open(op_csv, "rb") as source:
        rdr = csv.reader(source)
        header = next(rdr)
        while i < len(range):
            op_csv_list.append(strings[0] + "_" + str(i) + ".csv")
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
    print "Calculating polarity"
    extract_sentiment(ip_csv, op_csv)
    print "-" * 100
    print "Splitting files polarity"
    range = [100, 200, 400, 600, 1000, 1500, 2000, 10000]
    op_csv_list = split_files(op_csv,range)

    for file in op_csv_list:
        print "Staring with kmeans++ for ", file
        kmeans_clustering(file)
        print "-" * 100
        print "Results stats for ", file
        print_stats(file, 4)
        print "-" * 100
        command = raw_input("Continue?")
        if (command == 'n'):
            break

    '''
    print "Staring with agglomerative_clustering with ward"
    agglomerative_clustering(op_csv, 'ward', None)
    print_stats(op_csv, 5)
    print "-" * 100
    '''
