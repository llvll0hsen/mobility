import json
import os

import pandas
import pickle
import numpy as np
from PreprocessTags import *
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import gensim, logging


def format_tags(tags):
    tags_string = ','.join(tags)
    tags_string = tags_string.lower()
    tags_string = cond_lemmatize(tags_string)
    tags_string = tags_string.replace(' ', '_')
    tags_string = tags_string.replace('/', ' ')
    return tags_string

def extract_tags(records):
    for r in records:
        tags = r["tags"]
        categories = r["categories"]
        tags.extend(categories)
        tags = list(set(tags))
        tags = format_tags(tags)

if __main__ == "__model__":

    print "train model"
    model = gensim.models.KeyedVectors.load_word2vec_format(os.path.join("data","GoogleNews-vectors-negative300.bin", binary=True)
    
    f = open('geosegmentation.venues.json','rb')
    records = json.load(f)


