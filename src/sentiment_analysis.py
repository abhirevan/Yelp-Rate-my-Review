import csv
from textblob import TextBlob
import argparse


def extract_sentiment(ip_csv, op_csv):
    with open(ip_csv, "rU") as source, open(op_csv, "wb") as result:
        rdr = csv.reader(source)
        wtr = csv.writer(result)
        wtr.writerow(next(rdr) + ["Polarity"])
        i = 0
        for r in rdr:
            blob = TextBlob(r[1].decode("utf8"))
            wtr.writerow(r + [blob.sentiment.polarity])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Find sentiment analysis for csv',
    )
    parser.add_argument(
        'ip_csv',
        type=str,
        help='The input csv file to extract.',
    )
    args = parser.parse_args()
    ip_csv = args.ip_csv
    op_csv = '{0}_senti.csv'.format(ip_csv.split('.csv')[0])
    print "-" * 100
    print "Calculating polarity"
    extract_sentiment(ip_csv, op_csv)
    print "-" * 100
