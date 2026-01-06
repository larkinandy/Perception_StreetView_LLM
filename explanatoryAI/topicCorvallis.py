# topicCorvallis.py #
# Author: Andrew Larkin
# Summary: create a topic model with 25 vectors and get predictions for all image comparisons

# import libraries
from Topic import Topic
import pandas as ps

# initialize an intance of the Toic class
topicModeler = Topic("/mnt/h",debug=True)

# create the topic model
topicModeler.createTopicModel()

# load model IF it has already been created 
#topicModeler = Topic("/mnt/h/",modelType = 'reducedModel10',loadWordKeys=True,debug=True)

# get topics for corvallis comparisons in batch size of 100000 (for a RTX 5090 GPU)
topicModeler.getPostTopics(10000,'/mnt/h/CorvallisComparisons.csv','/mnt/h/CorvallisTopics.csv')

# end of topicCorvallis.py #