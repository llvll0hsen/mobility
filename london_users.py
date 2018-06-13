import os
import sys
from dateutil.parser import parse
import datetime
from collections import defaultdict,Counter
import dill
import operator

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pylab as plt

from util import output_path_files, output_path_plots,connect_mongodb, reverse_geo_mongodb,mobility_path,census_data_fpath

output_path_plots = os.path.join(output_path_plots,"mobility")
output_path_files = os.path.join(output_path_files,"mobility")
month ="august"
def map_user_group():
    '''
    map users to a deprivation group
    {user: group id}
    '''
    fpath = os.path.join(output_path_files, "user_top_antenna_london_only_deprivation_{0}_00_04.txt".format(month))
    df = pd.read_csv(fpath,usecols=["user_id","deprivation_rank"])
    print df.head()
    user_dep =  df.set_index('user_id').to_dict()['deprivation_rank']
    
    
    dill.dump(user_dep, open(os.path.join(output_path_files,"user_groups_dict.dill"),"wb"))
    #return group_userids 

def map_group_users():
    '''
    split users to deprivation groups
    {group_id: list of users}
    '''
    df_london = pd.read_csv(os.path.join(output_path_files, "user_top_antenna_london_only_deprivation_{0}_00_04.txt".format(month)))
    gp = df_london.groupby("deprivation_rank")["user_id"]
    groups = gp.groups.keys()
    group_userids = {} 
    for g in groups:
        group_userids[g] = gp.get_group(g).tolist()
    dill.dump(group_userids, open(os.path.join(output_path_files,"group_users_dict.dill"),"wb"))
    

def antenna_deprivation():
    antenna_lsoa = dill.load(open(os.path.join(output_path_files,"antenna_lsoa_london_only_new.dill"),"rb"))
    df = pd.read_excel(os.path.join(census_data_fpath,"london","deprivation_london.xls"),sheet_name="IMD 2015")
    col_names = ["LSOA code (2011)","LSOA name (2011)","Local Authority District code (2013)","Local Authority District name (2013)","IMD Decile (where 1 is most deprived 10% of LSOAs)","IMD Rank (where 1 is most deprived)"]
    df = df[col_names].dropna()
    values = {}
    for row in df.itertuples():
        area_name, value = row[1],row[5]
        values[area_name.lower()] = int(value)
    aid_dep_rank = {}
    for aid, lsoa in antenna_lsoa.iteritems():
        aid_dep_rank[aid] = values[lsoa.lower()] 
    print len(aid_dep_rank)
    dill.dump(aid_dep_rank,open(os.path.join(output_path_files,"antenna_depr.dill"),"wb"))

def split_social_groups(df_london):
    df = pd.read_excel(os.path.join(census_data_fpath,"london","deprivation_london.xls"),sheet_name="IMD 2015")
    col_names = ["LSOA code (2011)","LSOA name (2011)","Local Authority District code (2013)","Local Authority District name (2013)","IMD Decile (where 1 is most deprived 10% of LSOAs)","IMD Rank (where 1 is most deprived)"]
    df = df[col_names].dropna()
    values = {}

    for row in df.itertuples():
        area_name, value = row[1],row[5]
        values[area_name.lower()] = int(value)


    antenna_lsa = dill.load(open(os.path.join(output_path_files,"antenna_lsoa_london_only.dill"),"rb"))
    for aid, lsa in antenna_lsa.iteritems():
        try:
            rank = values[lsa.lower()]
            df_london.ix[df_london["top_antenna"]==aid,"deprivation_rank"] = rank
            df_london.ix[df_london["top_antenna"]==aid,"lsoa"] = lsa
        except Exception as err:
            df_london.ix[df_london["top_antenna"]==aid,"deprivation_rank"] = None
            df_london.ix[df_london["top_antenna"]==aid,"lsoa"] = lsa

    df_london.to_csv(os.path.join(output_path_files, "user_top_antenna_london_only_deprivation_{0}_00_04.txt".format(month)),index=False)
    print df_london.head()

if __name__ == "__main__":
    f =  open(os.path.join(output_path_files, "user_top_antenna_london_only_{0}_00_04.txt".format(month)),"rb")
    df = pd.read_csv(f)
#    split_social_groups(df)
#    antenna_deprivation()
    map_group_users()
    map_user_group()
    
