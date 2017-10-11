import os
import json
from collections import defaultdict

import numpy as np
from pymongo import MongoClient
    
from util import output_path_files,reverse_geo_mongodb

def ne_dist_cats(record,write_to_file=False):
    collection,client = connect_monogdb()
    f1 = open(os.path.join(output_path_files,"ne_cats.txt"),"wb")
    f1.writelines("neighborhood,categories")
    
    f2 = open(os.path.join(output_path_files,"dist_cats.txt"),"wb")
    f2.writelines("district,categories")
    
    
    ne_cats = defaultdict(set)
    dist_cats = defaultdict(set)

    for r in records:
        lon,lat = r['geolocation']['coordinates']
        cats = set(r['categories'])
        temp = reverse_geo_mongodb(lat,lon,collection)
        if temp:
            dist, ne = temp
#            print temp
            dist_cats[ne].update(cats)
            ne_cats[ne].update(cats)
    if write_to_file:
        for ne, cats in ne_cats.iteritems():
            cats_all = ":".join(cats)
            f1.writelines("\n{0},{1}".format(ne,cats_all))
        
        for dist, cats in dist_cats.iteritems():
            cats_all = ":".join(cats)
            f2.writelines("\n{0},{1}".format(ne,cats_all))
        f1.close()
        f2.close()
        client.close()
    return ne_cats, dist_cats

def tf_idf(data):
    pass

if __name__ == '__main__':
    f = open('geosegmentation_venues_170904.json','rb')
    records = json.load(f)
    ne_dist_cats(records,True)
