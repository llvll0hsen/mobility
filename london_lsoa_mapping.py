
import traceback
import os
import sys

import pandas as pd

from util import output_path_files, output_path_plots,connect_mongodb, reverse_geo_mongodb,census_data_fpath
import dill

def antenna_loc_todict():
    """
    create a dictionary -> {antenna_id -> (lat,lon)}
    """

    r = {}
    f =  open("data/location_withgps.txt","rb")
    for line in f:
        a = line.split("\t")
        r[a[0]] = (a[3],a[4])
    dill.dump(r,open(os.path.join(output_path_files,"mobility","antenna_loc.dill"),"wb"))
    return r

def find_missing_lsoas():
    """
    find lsoas with no antenna detected. 
    the file deprivation_london.xls can be downloaded from here:
    https://data.london.gov.uk/dataset/indices-of-deprivation-2015
    """

    #df = pd.read_excel(os.path.join(census_data_fpath,"london","deprivation_london.xls"),encoding='iso-8859-1', sheet_name="Sub domains")
    df = pd.read_csv(os.path.join(census_data_fpath, "london", "deprivation_london_sd.csv"),delimiter=";")
    #print df.columns.values
    lsoas_all = df["LSOA code (2011)"].tolist()
    lsoas = dill.load(open(os.path.join(output_path_files,"mobility","antenna_lsoa_london_only.dill"),"rb"))
    print len(set(lsoas.values()))
    temp = list(set(lsoas_all) - set(lsoas.values()))
    print 'total number of lsoas: {0}'.format(len(lsoas_all))
    print "number of missing lsoas: {0} {1}".format(len(temp), float(len(temp))/len(lsoas_all))

def mapping(antenna_loc):
    collection,client = connect_mongodb()
    r = {}
    r_lsoa = {}
    i = 0 
    for aid,loc in antenna_loc.iteritems():
        lon,lat = loc
        print lon, lat
        try:
            ne = reverse_geo_mongodb(float(lat),float(lon),collection)
            #if there is a match
            if ne:
                r[aid] = (lon,lat)
                r_lsoa[aid] = ne[0] #ne=(lsoa code, lsoa name)
                i+=1
        except Exception as err:
            traceback.print_exc()
            pass
    client.close()
    dill.dump(r,open(os.path.join(output_path_files,"mobility","antenna_loc_london_only.dill"),"wb"))
    dill.dump(r_lsoa, open(os.path.join(output_path_files,"mobility","antenna_lsoa_london_only.dill"),"wb"))
    
if __name__ == '__main__':
#    antenna_loc = antenna_loc_todict()
#    antenna_loc = dill.load(open(os.path.join(output_path_files,"mobility","antenna_loc.dill"),"rb"))
#    mapping(antenna_loc)
    find_missing_lsoas()
