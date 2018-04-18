import os
import dill
from pymongo import MongoClient
import pandas as pd

output_path = os.path.join(os.getcwd(),'output')
output_path_files  = os.path.join(os.getcwd(),'output','files')
output_path_plots = os.path.join(os.getcwd(),'output','plots')
census_data_fpath = os.path.join(os.getcwd(),'data','census')
mobility_path = os.path.join(os.getcwd(),'data','mining_mobility')

def get_antennas_lsoa():
    return dill.load(open(os.path.join(output_path_files,"mobility","antenna_lsoa_london_only_new.dill"),"rb"))
def connect_mongodb():
    mongo_client = MongoClient("localhost")
#    mongo_db = mongo_client.bcn
#    mongo_collection = mongo_db.barris
    mongo_db = mongo_client.london
#    mongo_collection = mongo_db.boroughs
    mongo_collection = mongo_db.lsoa2
#    print mongo_collection.find_one()
    return mongo_collection, mongo_client


def reverse_geo_mongodb(lat,lon, collection):
#    print lat,lon
    r = collection.find_one({"geometry":{"$geoIntersects":{"$geometry":{"type":"Point","coordinates":[lon,lat]}}}})
#    print "*",r
    if r:
#        print r
        d = r["properties"]["LSOA11CD"]
        b = r["properties"]["LSOA11NM"].encode("utf-8").lower()
#        d = r["properties"]["name"].encode("utf-8").lower()
#        b = r["properties"]["name"].encode("utf-8").lower()
#        d = r["properties"]["N_Distri"].encode("utf-8").lower()
#        b = r["properties"]["N_Barri"].encode("utf-8").lower()
#        print (d,b)
#        sys.exit()
        result = (d,b)
    else:
        result = None
    return result

def get_depr_factor(factor_name):
    df = pd.read_excel(os.path.join(census_data_fpath,"london","deprivation_london.xls"),sheet_name="IMD 2015")
    col_names = ["LSOA code (2011)",factor_name]
#    col_names = ["LSOA name (2011)","Children and Young People Sub-domain Decile (where 1 is most deprived 10% of LSOAs)","Adult Skills Sub-domain Decile (where 1 is most deprived 10% of LSOAs)","Geographical Barriers Sub-domain Decile (where 1 is most deprived 10% of LSOAs)","Wider Barriers Sub-domain Decile (where 1 is most deprived 10% of LSOAs)",
#            "Indoors Sub-domain Decile (where 1 is most deprived 10% of LSOAs)","Outdoors Sub-domain Decile (where 1 is most deprived 10% of LSOAs)"]
    df = df[col_names].dropna()
    df.ix[:,"LSOA code (2011)"] = df["LSOA code (2011)"].str.lower()
    df = df.set_index("LSOA code (2011)")
    dpr_dict = df.to_dict()[factor_name]
    
    return dpr_dict

def get_lsoa_list():
    df = pd.read_excel(os.path.join(census_data_fpath,"london","deprivation_london.xls"),sheet_name="Sub domains")
    col_names = ["LSOA code (2011)"]
    df = df[col_names].dropna()
    lsoas = df["LSOA code (2011)"].tolist()
#    dill.dump(lsoas,open("lsoa_list.dill","wb")) 

#    lsoas = dill.load(open(os.path.join(output_path_files,"lsoa_vanues.dill"),"rb"))
#    print lsoas.keys()

def get_depr_subdomain(factor_name):
    df = pd.read_excel(os.path.join(census_data_fpath,"london","deprivation_london.xls"),sheet_name="Sub domains")
    col_names = ["LSOA code (2011)",factor_name]
#    col_names = ["LSOA name (2011)","Children and Young People Sub-domain Decile (where 1 is most deprived 10% of LSOAs)","Adult Skills Sub-domain Decile (where 1 is most deprived 10% of LSOAs)","Geographical Barriers Sub-domain Decile (where 1 is most deprived 10% of LSOAs)","Wider Barriers Sub-domain Decile (where 1 is most deprived 10% of LSOAs)",
#            "Indoors Sub-domain Decile (where 1 is most deprived 10% of LSOAs)","Outdoors Sub-domain Decile (where 1 is most deprived 10% of LSOAs)"]
    df = df[col_names].dropna()
    df.ix[:,"LSOA code (2011)"] = df["LSOA code (2011)"].str.lower()
    df = df.set_index("LSOA code (2011)")
    dpr_dict = df.to_dict()[factor_name]
    
    return dpr_dict
if __name__ == '__main__':
    get_lsoa_list()
