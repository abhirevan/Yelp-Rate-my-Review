import numpy as np
import pandas as pd


def print_avg_polarity_stars(ip_csv):
    review_csv = pd.read_csv(ip_csv)
    grouped = review_csv.groupby('stars')
    avg = {}
    median = {}
    for name,group in grouped:
        avg[name]=np.mean(group['Polarity'])
        median[name]=np.median(group['Polarity'])

    print "Mean"
    print avg
    print "Median"
    print median




if __name__ == '__main__':
    print_avg_polarity_stars('data\input\\results\wSubjectivity\yelp_academic_dataset_review_ext_senti.csv')
