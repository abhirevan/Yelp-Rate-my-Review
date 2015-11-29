import csv
from numpy import histogram


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


def business_stats(categories, category_count):
    ip_csv = "data\input\yelp_academic_dataset_business_ext.csv"
    with open(ip_csv, "rb") as source:
        rdr = csv.reader(source)
        next(rdr)
        # c = 0
        for r in rdr:
            cat = r[0]
            items = cat.split(',')
            for i in items:
                i = i.lstrip()
                if category_count.has_key(i):
                    category_count[i] = category_count[i] + 1
                else:
                    category_count[i] = 1
                    categories.append(i)
                    # print categories
                    # print category_count


if __name__ == '__main__':
    count_ratings = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    length = []

    review_stats(count_ratings, length)
    print "Review Stats"
    print ('-' * 100)
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

    hist,bin_edges = histogram(a=length,bins=20)

    print hist
    print bin_edges


    '''
    print "Business Stats"
    print ('-' * 100)
    categories = []
    category_count = {}
    business_stats(categories, category_count)

    print "Number of categories", len(categories)
    print "Reviews per category:"

    for c in categories:
        print c + "?" + str(category_count[c])
    '''
