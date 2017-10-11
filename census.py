import os
import sys
from collections import defaultdict

import pandas as pd

from util import census_data_fpath,output_path_files,connect_monogdb


def get_ne_coords(ne,collection):
    print ne.decode("iso8859_15")
    r = collection.find({"properties.N_Barri": ne.decode("iso8859_15")})
    r =  r.next()
    coords = r["geometry"]["coordinates"]
    return coords[0]
def rename_ne(df):
    """
    rename Barris names to match the mongodb query
    """
    r1 = df["Barris"][10].split("-")[0].strip()
    r2 = df["Barris"][11].split("-")[0].strip()
    df["Barris"].replace(df["Barris"][10],r1,inplace=True)
    df["Barris"].replace(df["Barris"][11],r2,inplace=True)
    return df

if __name__ == "__main__":
#    f = open(os.path.join(census_data_fpath,"income_2015.csv"))
    f = open(os.path.join(census_data_fpath,"immigration_rate_2016.csv"))

    df = pd.read_csv(f,delimiter=",")
#    df = df[1:-1]
    df = df[:-1]
    df = rename_ne(df)


    collection, client = connect_monogdb()
    fname = "immigration_rate"
    ne_val = defaultdict(float)
    
    fpath = os.path.join(output_path_files,"census_{0}_coords_vals.txt".format(fname)) 

    f =  open(fpath, "wb")
#    f.writelines("neighborhood,lan,lon,val")
    f.writelines("neighborhood,val")

    for row in df.itertuples():
        print row
#        ne, val = row[2].split(".")[1].strip(), row[3]
        ne, val = row[3].split(".")[1].strip(), row[4]
        print ne
#        coords = get_ne_coords(ne,collection)
#        for coord in coords:
        ne =  ne.decode("iso8859_15").encode("utf-8")
        f.writelines("\n{0};{1}".format(ne,val))
    
    f.close()
    client.close()
        



