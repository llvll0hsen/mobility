import os
import sys
from dateutil.parser import parse
import datetime
from collections import defaultdict,Counter
import dill
import operator
import random

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from scipy.sparse import lil_matrix
from sklearn.metrics.pairwise import cosine_similarity

from util import output_path_files, output_path_plots,connect_monogdb, reverse_geo_mongodb,mobility_path,census_data_fpath

output_path_plots = os.path.join(output_path_plots,"mobility")
output_path_files = os.path.join(output_path_files,"mobility")
month ="august"

def boxplots(data1,data2,title):
#    print data1
#    print data2
    plt.figure() 
    ax = plt.subplot()
#    a = ax.boxplot([data1,data2] ,0,"",labels=["poor","rich"])
    a = ax.boxplot([data1,data2] ,labels=["poor","rich"])#, positions=[3,6])
    ax.set_title(title)
    ax.tick_params(direction='out')
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False) 
    ax.get_xaxis().tick_bottom()  
    ax.get_yaxis().tick_left()
#    ax.axis('tight')
    ax.margins(0.5)
    plt.savefig(os.path.join(output_path_plots,"poor_rich_{0}.png".format(title)),bbox_inches='tight')
    plt.close()

def plot_cdf(data,title):
    plt.figure() 
    ax = plt.subplot()
    for group,record in data.iteritems():
        print group
        hist = Counter(record)
        n = float(sum(hist.values()))
        normalized_count = {day: freq/n for day,freq in hist.iteritems()}
        sorted_pk = sorted(normalized_count.iteritems(), key=operator.itemgetter(0),reverse=True)
        x = [i[0] for i in sorted_pk]
        y = np.cumsum([i[1] for i in sorted_pk])
        a = ax.plot(x,y ,lw=5., alpha=0.6,label=group)
    
    ax.tick_params(direction='out')
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False) 
    ax.get_xaxis().tick_bottom()  
    ax.get_yaxis().tick_left()
    ax.set_xlabel("{0} d".format(title),fontsize=25)
    ax.set_ylabel("x>=d",fontsize=25)
    ax.axis('tight')
#    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.margins(0.05)
    ax.legend()
    plt.savefig(os.path.join(output_path_plots,"cdf_{0}.png".format(title)),bbox_inches='tight')
    plt.close()

def corr(a,b):
    #correlation between bboxdiagonal and gyration
    bbox = []
    gyration = []
    for user in a.iterkeys():
        bbox.append(np.median(a[user]))
        gyration.append(np.median(b[user]))

    print np.corrcoef(bbox, gyration)

def group_users():
    poor_index = range(4)
    rich_index = range(27,31)
    middle1_index = range(4,21)
    middle2_index = range(21,27)
    
    df_london = pd.read_csv(os.path.join(output_path_files, "user_top_anthena_london_only_economicRank_{0}_00_04.txt".format(month)))
    
#    print df_london.columns
    rich_users = df_london[df_london["economic_rank"].isin(rich_index)]["user_id"].unique().tolist()
    poor_users = df_london[df_london["economic_rank"].isin(poor_index)]["user_id"].unique().tolist()
    middle1_users = df_london[df_london["economic_rank"].isin(middle1_index)]["user_id"].unique().tolist()
    middle2_users = df_london[df_london["economic_rank"].isin(middle2_index)]["user_id"].unique().tolist()
    
#    poor_users = random.sample(poor_users, len(rich_users))
    print len(rich_users) 
    print len(poor_users)
    
    return rich_users,poor_users,middle1_users, middle2_users

def gyration_bbox_compare(rich_users,poor_users):
    user_gyration = dill.load(open("user_gyration.dill","rb"))
    user_bbdiagonal = dill.load(open("bbdiagonal.dill","rb"))
    
    poor_gyration = []
    rich_gyration = []

    poor_bbdiagonal = []
    rich_bbdiagonal = []
    func = np.mean 
    for i in poor_users:
        poor_gyration.append(func(user_gyration[i]))
        poor_bbdiagonal.append(func(user_bbdiagonal[i]))

    for i in rich_users:
        rich_gyration.append(func(user_gyration[i]))
        rich_bbdiagonal.append(func(user_bbdiagonal[i]))
   
    gyration = {}
    bboxdiag = {}

    gyration["poor"] = poor_gyration
    gyration["rich"] = rich_gyration
    plot_cdf(gyration,"gyration_downsample")
    
    bboxdiag["poor"] = poor_bbdiagonal
    bboxdiag["rich"] = rich_bbdiagonal
    plot_cdf(bboxdiag,"bboxdiag_downsample")
    boxplots(poor_gyration,rich_gyration,"gyration_downsample") 
    boxplots(poor_bbdiagonal,rich_bbdiagonal,"bbdiagonal_downsample") 
    corr(user_gyration,user_bbdiagonal)

def user_anthena_diversity(rich_users, poor_users):
    f_user_anthena =  open(os.path.join(output_path_files, "user_anthena_{0}_all.txt".format(month)),"rb")
    user_anthenas = defaultdict(set)
    for row in f_user_anthena:
        user_id,anthena_id,duration,date,weekday,time_slice = row.strip().split(",")
        user_anthenas[user_id].add(anthena_id)
    
    
#    dill.dump(user_anthenas, open(os.path.join(output_path_files,"user_anthena_august.dill"),"wb"))
    
#    sys.exit()
    poor_rich_anthena_count = defaultdict(list)
    for i in rich_users:
        poor_rich_anthena_count["rich"].append(len(user_anthenas[i]))

    for i in poor_users:
        poor_rich_anthena_count["poor"].append(len(user_anthenas[i]))

    boxplots(poor_rich_anthena_count["poor"], poor_rich_anthena_count["rich"],"anthena_count_downsample")
    
def users_neighbourhood_diversity(rich_users, poor_users):
    user_anthenas = dill.load(open(os.path.join(output_path_files,"user_anthena_august.dill"),"rb"))
    anthena_ne = dill.load(open(os.path.join(output_path_files,"anthena_ne_london_only.dill"),"rb"))
    
    users_ne = defaultdict(set)
    for user in rich_users:
        nes = set()
        for i in user_anthenas[user]:
            try:
                nes.add(anthena_ne[i]) 
            except Exception as err:
#                print err
                pass
        users_ne["rich"].add(len(nes))


    for user in poor_users:
        nes = set()
        for i in user_anthenas[user]:
            try:
                nes.add(anthena_ne[i]) 
            except Exception as err:
                pass
        users_ne["poor"].add(len(nes))

    boxplots(list(users_ne["poor"]),list(users_ne["rich"]),"ne_count_downsample")

def users_trajectory_to_file(user_groups):
    user_anthenas = dill.load(open(os.path.join(output_path_files,"user_anthenas_august_all.dill"),"rb"))
    anthena_loc = dill.load(open(os.path.join(output_path_files,"anthena_loc_london_only.dill"),"rb"))
    for group, user_ids in user_groups.iteritems():
        fpath = os.path.join(output_path_files,"trajectories_{0}_{1}_{2}.txt".format(month,group,"all"))
        with open(fpath,"wb") as f:
            f.writelines("lon,lat\n")
            for i in user_ids:
                anthenas = user_anthenas[i]
                anthenas = [k for k,v in anthenas.iteritems() if v>10]
                for a in anthenas:
                    try:
                        lon,lat = anthena_loc[a]
                        f.writelines("{0},{1}".format(lon,lat))
                    except Exception as err:
                        pass


def users_trajectory_similariy(user_groups):
    user_anthenas = dill.load(open(os.path.join(output_path_files,"user_anthenas_august_all.dill"),"rb"))
    anthena_loc = dill.load(open(os.path.join(output_path_files,"anthena_loc_london_only.dill"),"rb"))
    anthena_ids = sorted(anthena_loc.keys())
    anthena_indices = dict(zip(anthena_ids,range(len(anthena_ids))))
    group_names = sorted(user_groups.keys())
    print group_names
    user_count = sum([len(v) for v in user_groups.itervalues()])
#    print user_count, len(anthena_ids)

#    user_matrix = np.zeros((user_count, len(anthena_ids)))
    user_matrix = lil_matrix((user_count, len(anthena_ids)),dtype=np.int8) 
    print user_matrix.shape
    user_idx = 0
    user_indices = {}
    for gn in group_names:
        user_ids = user_groups[gn]
        for uid in user_ids:
            anthenas_count = user_anthenas[uid]
            for aid, count in anthenas_count.iteritems():
                try:
                    anthena_idx = anthena_indices[aid] 
                    user_matrix[user_idx, anthena_idx] = count
                except Exception as err:
                    #anthena outside london
                    pass
            user_indices[user_idx] = uid
            user_idx+=1
    

    similarities = cosine_similarity(user_matrix)
    plt.figure()
    fig, ax = plt.subplots()
    ax.imshow(similarities, interporlation="nearest")
    plt.savefig(os.path.join(output_path_plots,"similarity.pdf"), bbox_inches="tight")
#    dill.dump((user_matrix,user_indices),open(os.path.join(output_path_files,"trajectory_matrix.dill"),"wb"))

            

   

if __name__ == "__main__":
    users = {}
    rich_users, poor_users, middle1_users, middle2_users = group_users()
    
    users["rich"] = rich_users
    users["poor"] = poor_users
    users["middle1"] = middle1_users
    users["middle2"] = middle2_users
#    users_trajectory_similariy(users)
    users_trajectory_to_file(users)
#    gyration_bbox_compare(rich_users,poor_users)
#    user_anthena_diversity(rich_users, poor_users)
#    users_neighbourhood_diversity(rich_users, poor_users)
#    user_anthena_diversity(rich_users, poor_users)
