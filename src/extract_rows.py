import csv
import argparse


def extract_rows(ip_n, ip_csv, op_csv):
    with open(ip_csv, "rb") as source, open(op_csv, "wb") as result:
        rdr = csv.reader(source)
        wtr = csv.writer(result)
        i = 0
        for r in rdr:
            if i <= ip_n:
                wtr.writerow((r))
                i += 1
            else:
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Find sentiment analysis for csv',
    )
    parser.add_argument(
        'ip_csv',
        type=str,
    )
    parser.add_argument(
        'ip_n',
        type=int,
    )
    args = parser.parse_args()
    ip_n = args.ip_n
    ip_csv = args.ip_csv
    op_csv = '{0}_ext.csv'.format(ip_csv.split('.csv')[0])

    extract_rows(ip_n, ip_csv, op_csv)
