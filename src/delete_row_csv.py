import csv
import argparse


def extract_rows(ip_csv_path, op_csv_path):
    print "in extract_rows"
    with open(ip_csv_path,"rb") as source:
        rdr= csv.reader( source )
        with open(op_csv_path,"wb") as result:
            wtr= csv.writer( result )
            for r in rdr:
                if len(r[0])>0:
                    wtr.writerow( (r) )



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
    op_csv = '{0}_ext_nz.csv'.format(ip_csv.split('.csv')[0])
    extract_rows(ip_csv,op_csv)