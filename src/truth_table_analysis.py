__author__ = 'priyanka'
import numpy
from pandas import DataFrame

neg_12stars = pd.read_csv('data/input/yelp_academic_dataset_review_Doctors_senti_strip_clusn0.csv')
pos_45stars = pd.read_csv('data/input/yelp_academic_dataset_review_Doctors_senti_strip_clusp0.csv')

truth_table = numpy.zeros((5, 5))


for i, row in enumerate(neg_12stars.values):
    truth_table[row[2]-1, row[4]-1] += 1
    #print DataFrame(truth_table)
    #print

for j, rowj in enumerate(pos_45stars.values):
    #print rowj[2]
    truth_table[rowj[2]-1, rowj[4]-1] += 1
    #print DataFrame(truth_table)
    #print
print DataFrame(truth_table)
