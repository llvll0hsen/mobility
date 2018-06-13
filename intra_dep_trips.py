import os
import sys
from dateutil.parser import parse
import datetime
from collections import defaultdict,Counter
import dill

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pylab as plt
from matplotlib import cm


from util import output_path_files, output_path_plots,connect_mongodb, reverse_geo_mongodb, get_antennas_lsoa

output_path_plots = os.path.join(output_path_plots,"mobility")
output_path_files = os.path.join(output_path_files,"mobility")
month ="august"

def plot(matrix,labels,fname):
    for row_id, label in labels.iteritems():
        temp_matrix = matrix[row_id,:,:]
        row_sum = temp_matrix.sum(1)
        temp_matrix /= row_sum[:,np.newaxis]
        print temp_matrix.shape
        fig = plt.figure() 
        ax = plt.subplot()

        cax = ax.imshow(temp_matrix, interpolation='nearest', cmap=plt.cm.Blues, vmin=0,vmax=0.4)
        fig.colorbar(cax)
        print output_path_plots
        plt.savefig("{0}.png".format(label))#os.path.join(output_path_plots,"dpr_trips","{0}.png".format(label)))
        plt.close()

def create_matrices():

    antenna_loc = dill.load(open(os.path.join(output_path_files,"antenna_loc_london_only.dill"),"rb"))
    antennas_lsoa = get_antennas_lsoa()
    user_home_antenna = pd.read_csv(os.path.join(output_path_files, "user_top_antenna_london_only_deprivation_{0}_00_04.txt".format(month)),usecols=["user_id","top_antenna"])
    
    user_home_antenna = user_home_antenna.dropna()
    user_home_antenna = user_home_antenna.set_index("user_id").to_dict()["top_antenna"]
    a = user_home_antenna.keys()
    
    antenna_dpr = dill.load(open(os.path.join(output_path_files,"antenna_depr.dill"),"rb"))
    antenna_ids = sorted(antenna_loc.keys())
    antenna_indices = dict(zip(antenna_ids,range(len(antenna_ids))))
    
    user_groups = dill.load(open(os.path.join(output_path_files,"group_users_dict.dill"),"rb"))
    group_names = sorted(user_groups.keys())#[:2]

    
    file_names  = os.listdir(os.path.join(output_path_files,"time_sliced"))#[:2] #["hour=all"] #["hour=00-04"] 
    file_names = [i for i in file_names if "time_user_antennas" in i]
    file_names_temp = [i.replace("_london_only.dill","") for i in file_names]
    
    week_aggr = defaultdict(float)
    ts_aggr = defaultdict(float)
    time_spans = {"hour=00-04":0,"hour=04-08":1,"hour=08-12":2,"hour=12-16":3,"hour=16-20":4,"hour=20-24":5}#[2:]
    time_labels = ["00-04","04-08","08-12","12-16","16-20","20-24"]#[2:]
    weekday_labels = {0:"Mon", 1:"Tue", 2:"Wed", 3:"Thu", 4:"Fri", 5:"Sat", 6:"Sun"}
    
    group_matrix_daily = np.zeros((len(time_spans),len(user_groups), len(user_groups))) #7 time slice
    group_matrix_weekly = np.zeros((len(weekday_labels),len(user_groups), len(user_groups))) #7 time slice
    for fn in file_names_temp:
        print fn
        dstr = fn.split("_")[-1]
        datetime_date = parse(dstr.split("=")[1]) 
        weekday = datetime_date.weekday()
        user_antennas = dill.load(open(os.path.join(output_path_files,"time_sliced","{0}_london_only.dill".format(fn)),"rb"))
        for i, gn in enumerate(group_names):
            user_ids = user_groups[gn]
            for uid in user_ids:
                home_antenna = user_home_antenna[uid]
                try:
                    del user_antennas["hour=all"]
                except Exception as err:
                    pass
                for time_slice, user_ant in user_antennas.iteritems():
                    print time_slice
#                    print user_ant 
                    ts_id = time_spans[time_slice]
                    antennas_count = user_ant[uid]
                    antennas_count.pop(home_antenna,None)
                    for aid, count in antennas_count.iteritems():
                        try:
                            aid_depr = int(antenna_dpr[aid])-1
                            antenna_idx = antenna_indices[aid] 
                            group_matrix_weekly[weekday,i, aid_depr] += count
                            group_matrix_daily[ts_id,i, aid_depr] += count

                        except Exception as err:
                            pass
    dill.dump(group_matrix_daily,open(os.path.join(output_path_files,"trip_matrix_daily.dill"),"wb"))
    dill.dump(group_matrix_weekly,open(os.path.join(output_path_files,"trip_matrix_weekly.dill"),"wb"))

if __name__ == '__main__':
#    create_matrices()
#    sys.exit()
    labels = {0:"00-04",1:"04-08",2:"08-12",3:"12-16",4:"16-20",5:"20-24"}
    matrix = dill.load(open(os.path.join(output_path_files,"trip_matrix_daily.dill"),"rb"))
    plot(matrix, labels,"daily")
#    labels = {0:"Mon", 1:"Tue", 2:"Wed", 3:"Thu", 4:"Fri", 5:"Sat", 6:"Sun"}
#    matrix = dill.load(open(os.path.join(output_path_files,"trip_matrix_weekly.dill"),"rb"))
#    plot(matrix,labels, "weekly")
