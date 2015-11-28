import csv


def review_stats(count_ratings, length):
    # print "in extract_rows"
    ip_csv = "data\input\yelp_academic_dataset_review_ext.csv"
    with open(ip_csv, "rb") as source:
        rdr = csv.reader(source)
        firstline = True
        for r in rdr:
            if firstline:  # skip first line
                firstline = False
                continue
            count_ratings[int(r[2])] += 1
            length.append(len(r[0]))


if __name__ == '__main__':
    count_ratings = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    length = []
    review_stats(count_ratings, length)
    print "total reviews", count_ratings[1] + count_ratings[2] + count_ratings[3] + count_ratings[4] + count_ratings[5]
    print "Review breakup per ratings"
    print "Review 1 star", count_ratings[1]
    print "Review 2 star", count_ratings[2]
    print "Review 3 star", count_ratings[3]
    print "Review 4 star", count_ratings[4]
    print "Review 5 star", count_ratings[5]

    length.sort()

    sum = 0.0
    for i in length:
        sum += i

    print "Min length: ", min(length), "Max length: ", max(length)
    print "Avg length: ", sum / len(length), "Median: ", length[len(length) / 2]

    
