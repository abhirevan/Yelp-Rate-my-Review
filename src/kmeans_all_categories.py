import csv
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
        i = 0
        for r in rdr:
            data.append([])
            data[i].append(float(r[3]))
            data[i].append(float(r[4]))
            i += 1
    data = np.array(data)  # .reshape(-1, 1)
    # print data
    centroids, labels = kmpp(data, 6, mode)
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
            y_pred.append(row[5])

    print confusion_matrix(y_true, y_pred)
    print precision_recall_fscore_support(y_true, y_pred, average='micro')


def read_business_categories(cat_file):
    categories = []
    cat_csv = pd.read_csv(cat_file)
    for i, row in enumerate(cat_csv.values):
        categories.append(row[0])
    return categories


def extract_category(category):
    op_csv = 'data\input\\results\BusinessCat\\yelp_academic_dataset_review_' + category + '.csv'

    business_data = pd.read_csv('data/input/yelp_academic_dataset_business.csv', index_col=('business_id'))
    review_data = pd.read_csv('data/input/yelp_academic_dataset_review_ext.csv', index_col=('business_id'))
    sample_business_ids = []
    for num, item in enumerate(business_data.index):
        if category in business_data.ix[num, 'categories']:
            sample_business_ids.append(item)

            # sample_business_ids = business_data[business_data['categories'].str.contains(cat) == True]
    # print sample_business_ids
    sample_business_data = business_data[business_data.index.isin(sample_business_ids)]
    sample_review_data = review_data[review_data.index.get_level_values(0).isin(sample_business_data.index)]
    # print sample_review_data
    df = pd.DataFrame(sample_review_data)
    df.to_csv(op_csv, cols=['sum'])
    return op_csv


def perform_multi_2_clustering(ip_csv):
    print "Extract Level 3 reviews for ", ip_csv
    ip_csv = strip_reviews(ip_csv)

    print "Splitting files polarity"
    #range = [100, 10000]
    range = [200, 400, 600, 10000]
    #range = [100, 200, 300, 400, 500, 10000]
    #range = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 10000]
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


if __name__ == '__main__':
    cat_file = "data\input\\results\BusinessCat\\business_cat.csv"
    categories = read_business_categories(cat_file)

    extracted_categories_file = []
    for cat in categories:
        extracted_categories_file.append(extract_category(cat))

    print extracted_categories_file

    for ip_csv in extracted_categories_file:
        perform_multi_2_clustering(ip_csv)
