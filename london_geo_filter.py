import traceback
import os
import sys

import pandas as pd

from util import output_path_files, output_path_plots,connect_mongodb, reverse_geo_mongodb,census_data_fpath
import dill
def anthena_loc_todict():
    r = {}
#    f =  open("cellcatalog_london.tsv","rb")#,delimiter="\t",error_bad_lines=False,warn_bad_lines=False)
    f =  open("data/location_withgps.txt","rb")#,delimiter="\t",error_bad_lines=False,warn_bad_lines=False)
    for line in f:
        a = line.split("\t")
        r[a[0]] = (a[3],a[4])

#    dill.dump(r,open(os.path.join(output_path_files,"mobility","anntena_loc_london_only.dill"),"wb"))
    dill.dump(r,open(os.path.join(output_path_files,"mobility","antenna_loc_new.dill"),"wb"))

def find_missing_lsoas():
    df = pd.read_excel(os.path.join(census_data_fpath,"london","deprivation_london.xls"),sheet_name="Sub domains")
    lsoas_all = df["LSOA code (2011)"].tolist()
    lsoas = dill.load(open(os.path.join(output_path_files,"mobility","antenna_lsoa_london_only_new.dill"),"rb"))
#    print lsoas.values()
    print len(set(lsoas.values()))
    temp = list(set(lsoas_all) - set(lsoas.values()))

#    print temp
    print len(lsoas_all) , len(temp)
def london_only_aid_loc():
    r = dill.load(open(os.path.join(output_path_files,"mobility","antenna_loc_london_only_new.dill"),"rb"))
    f = open("output/files/london_only_aid_loc.txt","wb")
    f.write("lon,lat")
    for aid,loc in r.iteritems():
        lon,lat = loc
        f.write("{0},{1}".format(lon,lat))

if __name__ == "__main__":
#    anthena_loc_todict()
#    london_only_aid_loc()
    find_missing_lsoas()
    antenna_loc = dill.load(open(os.path.join(output_path_files,"mobility","antenna_loc_new.dill"),"rb"))
    print len(antenna_loc)
    sys.exit()
    antenna_loc2 = dill.load(open(os.path.join(output_path_files,"mobility","antenna_loc.dill"),"rb"))
    aids = dill.load( open(os.path.join(output_path_files,"mobility","antenna_set.dill"),"rb"))
    a= aids - set(set(antenna_loc.keys()))
    print len(aids)
#    print aids
 
#    print set(antenna_loc.keys()) - set(antenna_loc2.keys())
    print len(a),len(antenna_loc2), len(antenna_loc)
    sys.exit()
    collection,client = connect_mongodb()
#    df =  pd.read_csv("cellcatalog.tsv",delimiter="\t",error_bad_lines=False,warn_bad_lines=False)
#    row_ids = []
#    for row in df.iterrows():
    r = {}
    r_lsoa = {}
#    del antenna_loc[""]
    i = 0 

    for aid,loc in antenna_loc.iteritems():
        lon,lat = loc
#        print aid,loc
#        lon,lat = row[1][[-2,-1]]  
        try:
            ne = reverse_geo_mongodb(float(lat),float(lon),collection)
            if ne:
    #            print ne
                r[aid] = (lon,lat)
                r_lsoa[aid] = ne[0]
                i+=1
        except Exception as err:
            traceback.print_exc()
            pass
#            row_ids.append(row[0])

#    df = df.drop(df.index[row_ids])
#    df.to_csv("cellcatalog_london.tsv",sep="\t",index=False)
    client.close()
    print i
    dill.dump(r,open(os.path.join(output_path_files,"mobility","antenna_loc_london_only_new.dill"),"wb"))
    dill.dump(r_lsoa, open(os.path.join(output_path_files,"mobility","antenna_lsoa_london_only_new.dill"),"wb"))


