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

    dill.dump(r,open(os.path.join(output_path_files,"mobility","anthena_loc_london_only.dill"),"wb"))

if __name__ == "__main__":
    anthena_loc_todict()
    sys.exit()
    collection,client = connect_monogdb()
    df =  pd.read_csv("cellcatalog.tsv",delimiter="\t",error_bad_lines=False,warn_bad_lines=False)
    row_ids = []
    for row in df.iterrows():

        lon,lat = row[1][[-2,-1]]  
        ne = reverse_geo_mongodb(lat,lon,collection)
        if not ne:
            row_ids.append(row[0])

    df = df.drop(df.index[row_ids])
    df.to_csv("cellcatalog_london.tsv",sep="\t",index=False)
    client.close()


