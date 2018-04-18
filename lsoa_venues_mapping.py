import os 
import sys
from collections import defaultdict
import json

import dill

from util import connect_mongodb,reverse_geo_mongodb,output_path_files

def mapping():
    f = open('london_comp.json','rb')
    records = json.load(f)
    collection,client = connect_mongodb()
    result = defaultdict(list)
    missing_venues =[]
    i = 0
    for r in records:
        lon,lat = r['geolocation']['coordinates']
        oid = r['_id']['$oid']
#        print lon,lat
        ne = reverse_geo_mongodb(lat,lon,collection)
#        print ne 
        try:
            lsoaid = ne[0]
            result[lsoaid].append(oid)
#            print lsoaid,result[lsoaid]
#            i+=1
        except Exception as err:
#            print err
            missing_venues.append(oid)
    dill.dump(result, open(os.path.join(output_path_files,"lsoa_vanues.dill"),"wb"))
    dill.dump(missing_venues, open(os.path.join(output_path_files,"missing_lsoa_vanues.dill"),"wb"))
    print len(missing_venues),len(set(missing_venues))
if __name__ == '__main__':
#    mapping()
#    sys.exit()
    lsoa_venues = dill.load(open(os.path.join(output_path_files,"lsoa_vanues.dill"),"rb"))
    missing_venues = dill.load(open(os.path.join(output_path_files,"missing_lsoa_vanues.dill"),"rb"))
    print len(missing_venues),len(lsoa_venues)
    keys = lsoa_venues.keys()
#    print lsoa_venues
    print keys[:10]
    print "E01002438" in lsoa_venues
    print lsoa_venues["E01002504"]
    print lsoa_venues["E01002438"]
#    print lsoa_venues["E01002438"]
