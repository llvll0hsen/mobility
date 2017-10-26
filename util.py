import os

from pymongo import MongoClient

output_path = os.path.join(os.getcwd(),'output')
output_path_files = os.path.join(os.getcwd(),'output','files')
output_path_plots = os.path.join(os.getcwd(),'output','plots')
census_data_fpath = os.path.join(os.getcwd(),'data','census')
mobility_path = "/home/bmja/Documents/mining-mobility"
def connect_monogdb():
    mongo_client = MongoClient("localhost")
#    mongo_db = mongo_client.bcn
#    mongo_collection = mongo_db.barris
    mongo_db = mongo_client.london
    mongo_collection = mongo_db.boroughs
    return mongo_collection,mongo_client

def reverse_geo_mongodb(lat,lon, collection):
    r = collection.find_one({"geometry":{"$geoIntersects":{"$geometry":{"type":"Point","coordinates":[lon,lat]}}}})
    if r:
        d = r["properties"]["name"].encode("utf-8").lower()
        b = r["properties"]["name"].encode("utf-8").lower()
#        d = r["properties"]["N_Distri"].encode("utf-8").lower()
#        b = r["properties"]["N_Barri"].encode("utf-8").lower()
        result = (d,b)
    else:
        result = None
    return result


