import traceback
import os
import sys

import pandas as pd

from util import output_path_files, output_path_plots,connect_monogdb, reverse_geo_mongodb
import dill
def anthena_loc_todict():
    r = {}
    f =  open("cellcatalog_london.tsv","rb")#,delimiter="\t",error_bad_lines=False,warn_bad_lines=False)
    for line in f:
        a = line.split("\t")
        r[a[3]] = (a[4],a[5])

    dill.dump(r,open(os.path.join(output_path_files,"mobility","anntena_loc_london_only.dill"),"wb"))

if __name__ == "__main__":
#    anthena_loc_todict()
#    sys.exit()
    antenna_loc = dill.load(open(os.path.join(output_path_files,"mobility","antenna_loc.dill"),"rb"))
    collection,client = connect_monogdb()
#    df =  pd.read_csv("cellcatalog.tsv",delimiter="\t",error_bad_lines=False,warn_bad_lines=False)
#    row_ids = []
#    for row in df.iterrows():
    r = {}
    r_lsoa = {}
    del antenna_loc[""]
    i = 0 

    for aid,loc in antenna_loc.iteritems():
        lon,lat = loc

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
    dill.dump(r,open(os.path.join(output_path_files,"mobility","anntena_loc_london_only2.dill"),"wb"))
    dill.dump(r_lsoa, open(os.path.join(output_path_files,"mobility","antenna_lsoa_london_only2.dill"),"wb"))


