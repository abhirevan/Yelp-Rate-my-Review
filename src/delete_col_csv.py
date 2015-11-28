import csv
import argparse

def extract_columns(ip_csv_path, op_csv_path):
    print "in extract_columns"
    with open(ip_csv_path,"rb") as source:
        rdr= csv.reader( source )
        with open(op_csv_path,"wb") as result:
            wtr= csv.writer( result )
            for r in rdr:
                wtr.writerow( (r[2], r[4], r[6]) )

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
            description='Convert Yelp Dataset Challenge data from JSON format to CSV.',
            )

    parser.add_argument(
            'ip_csv',
            type=str,
            help='The input csv file to extract.',
            )

    args = parser.parse_args()

    ip_csv = args.ip_csv
    op_csv = '{0}_ext.csv'.format(ip_csv.split('.csv')[0])
    extract_columns(ip_csv,op_csv)