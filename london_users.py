import os
import sys
from dateutil.parser import parse
import datetime
from collections import defaultdict,Counter
import dill
import operator

import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from util import output_path_files, output_path_plots,connect_monogdb, reverse_geo_mongodb,mobility_path,census_data_fpath

output_path_plots = os.path.join(output_path_plots,"mobility")
output_path_files = os.path.join(output_path_files,"mobility")
month ="august"
def ne_user_count(df):
    ne_size = df.groupby("neighborhood").size().to_dict()
    ne_size = sorted(ne_size.items(),key=operator.itemgetter(1))
    dill.dump(ne_size, open(os.path.join(output_path_files,"ne_population.dill"),"wb"))


    fig,ax = plt.subplots()
    loc, count = zip(*ne_size)
    x = range(len(loc))
    ax.bar(x,count, align="center")
    ax.set_xticks(x)
    ax.set_xticklabels(loc,rotation=90)
    ax.tick_params(direction='out')
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False) 
    ax.get_xaxis().tick_bottom()  
    ax.get_yaxis().tick_left()
    fpath = os.path.join(output_path_plots,'user_ne_dist.pdf')
    plt.savefig(fpath,bbox_inches='tight')
    plt.close()

def split_rich_poor_users(df_london):
    month = "august"
    col_names = ["Area name","Gross Annual Pay, (2016)"]
    df = pd.read_excel(os.path.join(census_data_fpath,"london","london-borough-profiles.xlsx"),sheetname="Data")
    df = df[col_names].dropna()
#    df = remove_invald_regions(df)
    df_london.insert(loc=len(df_london.columns),column="economic_rank",value=0)

    values = {}
    temp = df[col_names]
    for row in temp.itertuples():
        area_name, value = row[1],row[2]
        try:
            values[area_name.lower()] = int(value)
        except Exception as err:#
            pass
            
#            values[area_name.lower()] = None
    del values["united kingdom"]
    del values["england"]
    del values["national comparator"]
    del values["london"]


    ne_richeness = sorted(values.items(),key=operator.itemgetter(1))
#    ne_richeness = [i for i in ne_richeness if i[1] ]
    for i, ne in enumerate(ne_richeness):
        print i,ne
        df_london.ix[df_london["neighborhood"]==ne[0],"economic_rank"] = i

    df_london.to_csv(os.path.join(output_path_files, "user_top_anthena_london_only_economicRank_{0}_00_04.txt".format(month)),index=False)

    fig,ax = plt.subplots()
    loc, count = zip(*ne_richeness)
    x = range(len(loc))
    ax.plot(x,count)
    ax.set_xticks(x)
    ax.set_xticklabels(loc,rotation=90)
    ax.tick_params(direction='out')
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False) 
    ax.get_xaxis().tick_bottom()  
    ax.get_yaxis().tick_left()
    fpath = os.path.join(output_path_plots,'dist_salary.pdf')
    plt.savefig(fpath,bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    f =  open(os.path.join(output_path_files, "user_top_anthena_london_only_{0}_00_04.txt".format(month)),"rb")
    df = pd.read_csv(f)
    ne_user_count(df)
    split_rich_poor_users(df)

