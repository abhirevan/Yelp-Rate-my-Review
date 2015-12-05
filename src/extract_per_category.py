__author__ = 'priyanka'
import sys
import pandas as pd

if __name__ == '__main__':
    reload(sys)
    sys.setdefaultencoding("utf-8")
    cat = raw_input("Category?")

    op_csv = 'data/input/yelp_academic_dataset_review_' + cat + '.csv'

    business_data = pd.read_csv('data/input/yelp_academic_dataset_business.csv', index_col=('business_id'))
    review_data = pd.read_csv('data/input/yelp_academic_dataset_review_ext.csv', index_col=('business_id'))
    sample_business_ids = []
    for num, item in enumerate(business_data.index):
        if cat in business_data.ix[num, 'categories']:
            sample_business_ids.append(item)

            # sample_business_ids = business_data[business_data['categories'].str.contains(cat) == True]
    # print sample_business_ids
    sample_business_data = business_data[business_data.index.isin(sample_business_ids)]
    sample_review_data = review_data[review_data.index.get_level_values(0).isin(sample_business_data.index)]
    # print sample_review_data
    df = pd.DataFrame(sample_review_data)
    df.to_csv(op_csv, cols=['sum'])
