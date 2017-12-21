import os
import sys
from dateutil.parser import parse
import datetime
from collections import defaultdict,Counter
import dill
import operator
import random

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pylab as plt
from scipy.sparse import lil_matrix
from sklearn.metrics.pairwise import cosine_similarity

from util import output_path_files, output_path_plots,connect_monogdb, reverse_geo_mongodb,mobility_path,census_data_fpath, get_antennas_lsoa
from antenna_analysis import heatmap_gini_overall, heatmap_gini_extreems

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
    if isinstance(data, dict):
        for group,record in data.iteritems():
#            print group
            hist = Counter(record)
            n = float(sum(hist.values()))
            normalized_count = {day: freq/n for day,freq in hist.iteritems()}
            sorted_pk = sorted(normalized_count.iteritems(), key=operator.itemgetter(0),reverse=True)
            x = [i[0] for i in sorted_pk]
            y = np.cumsum([i[1] for i in sorted_pk])
            a = ax.plot(x,y ,lw=5., alpha=0.6,label=group)
    else:
        hist = Counter(data)
        del hist[0]
#        print sorted(data)
        n = float(sum(hist.values()))
        normalized_count = {day: freq/n for day,freq in hist.iteritems()}
        sorted_pk = sorted(normalized_count.iteritems(), key=operator.itemgetter(0),reverse=True)
        x = [i[0] for i in sorted_pk]
        y = np.cumsum([i[1] for i in sorted_pk])
        print y
        a = ax.plot(x,y ,lw=5., alpha=0.6)

    ax.tick_params(direction='out')
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False) 
    ax.get_xaxis().tick_bottom()  
    ax.get_yaxis().tick_left()
    ax.set_xlabel("{0} d".format(title),fontsize=25)
    ax.set_ylabel("x>=d",fontsize=25)
    ax.axis('tight')
#    ax.set_xscale('log')
#    ax.set_yscale('log')
    ax.margins(0.05)
#    ax.legend()
    print title
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

def get_users_group():
    df_london = pd.read_csv(os.path.join(output_path_files, "user_top_anthena_london_only_deprivation_{0}_00_04.txt".format(month)))
    gp = df_london.groupby("deprivation_rank")["user_id"]
    groups = gp.groups.keys()
    group_userids = {} 
    for g in groups:
        group_userids[g] = gp.get_group(g).tolist()
        print g, len(group_userids[g])
    return group_userids 

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


def compute_avg_similarity(similarity_matrix, group_sizes, group_names):
    ngroups = len(group_names)
    avg_sim_matrix = np.zeros((ngroups, ngroups))
    ind1 = 0
    similarity_matrix = np.triu(similarity_matrix,k=1)
#    print group_sizes
    for i in xrange(ngroups):
        gs1 = group_sizes[i]
        ind2 = ind1
        for j in xrange(i,ngroups):
            gs2 = group_sizes[j]
#            print "({0},{1}),({2},{3})".format(ind1,ind1+gs1,ind2,ind2+gs2)
            temp_matrix = similarity_matrix[ind1:ind1+gs1,ind2:ind2+gs2]
            s =  np.mean(temp_matrix[np.nonzero(temp_matrix)])
            avg_sim_matrix[i, j] = s
            ind2+=gs2
        ind1+=gs1
    print avg_sim_matrix

def users_trajectory_similariy(user_groups):
#    user_anthenas = dill.load(open(os.path.join(output_path_files,"user_anthenas_august_all.dill"),"rb"))
    user_anthenas = dill.load(open(os.path.join(output_path_files,"user_anthenas_august_weekend_night.dill"),"rb"))
    anthena_loc = dill.load(open(os.path.join(output_path_files,"anthena_loc_london_only.dill"),"rb"))
    anthena_ids = sorted(anthena_loc.keys())
    anthena_indices = dict(zip(anthena_ids,range(len(anthena_ids))))
    group_names = sorted(user_groups.keys())
    group_sizes = [len(user_groups[gn]) for gn in group_names]

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
            anthenas_count ={k:v for k,v in anthenas_count.iteritems() if v>5}
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
    ax.matshow(similarities,cmap=plt.get_cmap("Blues"))
    i = 0
    for gs in group_sizes:
        ax.axhline(y=i+gs)
        ax.axvline(x=i+gs)
        i+=gs
    ax.margins(0.5)
#    plt.savefig(os.path.join(output_path_plots,"similarity_WE_min5.pdf"), bbox_inches="tight")
#    dill.dump((user_matrix,user_indices),open(os.path.join(output_path_files,"trajectory_matrix.dill"),"wb"))
    return similarities, group_sizes, group_names


def daily_mixing(user_groups):
    print "daily_mixing"
    antenna_loc = dill.load(open(os.path.join(output_path_files,"anthena_loc_london_only.dill"),"rb"))
    antennas_lsoa = get_antennas_lsoa()
    user_home_antenna = pd.read_csv(os.path.join(output_path_files, "user_top_anthena_london_only_deprivation_{0}_00_04.txt".format(month)),usecols=["user_id","top_anthena"])
    user_home_antenna.dropna()
    user_home_antenna = user_home_antenna.set_index("user_id").to_dict()["top_anthena"]

    antenna_ids = sorted(antenna_loc.keys())
    antenna_indices = dict(zip(antenna_ids,range(len(antenna_ids))))
    antenna_indices_r = dict(zip(range(len(antenna_ids)),antenna_ids))
    group_names = sorted(user_groups.keys())#[:2]
    group_sizes = [len(user_groups[gn]) for gn in group_names]

    print group_names
    user_count = sum([len(v) for v in user_groups.itervalues()])
    
    file_names  = os.listdir(os.path.join(output_path_files,"time_sliced"))#[:1] #["hour=all"] #["hour=00-04"] 
    
    weekday_labels = {0:"Mon", 1:"Tue", 2:"Wed", 3:"Thu", 4:"Fri", 5:"Sat", 6:"Sun"}
    week_aggr = defaultdict(float)
    ts_aggr = defaultdict(float)

    antenna_group_matrix = np.zeros((len(weekday_labels),len(antenna_ids),len(group_names)))
    for fn in file_names:
        dstr = fn.split(".")[0].split("_")[-1]

        datetime_date = parse(dstr.split("=")[1]) 
        weekday = datetime_date.weekday()
         
        user_antennas = dill.load(open(os.path.join(output_path_files,"time_sliced",fn),"rb"))
        for i, gn in enumerate(group_names):
            user_ids = user_groups[gn]
            for uid in user_ids:
                home_antenna = user_home_antenna[uid]
                for time_slice, user_ant in user_antennas.iteritems():
                    antennas_count = user_ant[uid]
                    #remove home anthena
                    antennas_count.pop(home_antenna,None)
                    for aid, count in antennas_count.iteritems():
                        try:
                            antenna_idx = antenna_indices[aid] 
                            antenna_group_matrix[weekday,antenna_idx, i] += count
                        except Exception as err:
                            pass
    weekday_ginis =  [] 
    weekday_top_antennas = {}
    for weekday in xrange(len(weekday_labels)):
        antenna_group_matrix[weekday,:,:]/= group_sizes 
        sums = antenna_group_matrix[weekday,:,:].sum(1)
        idx_nozero = np.where(sums!=0)[0]
        aidx_ginis = {i:gini_coeff(antenna_group_matrix[weekday,i,:]) for i in idx_nozero}
        weekday_ginis.append(aidx_ginis.values())
        fname = "day_{0}".format(weekday_labels[weekday])
        print "heatmap"
#        heatmap_gini_overall(aidx_ginis,antenna_indices_r,antennas_lsoa,idx_nozero,fname)
        heatmap_gini_extreems(aidx_ginis,antenna_indices_r,antennas_lsoa,idx_nozero,fname)
    
    print "boxplot"
    fig, ax = plt.subplots()
    ax.boxplot(weekday_ginis,showmeans=True)
#
    plt.savefig(os.path.join(output_path_plots,"mixing_weekday.png"), bbox_inches="tight")
    plt.close() 

def ts_mixing(user_groups):
    antenna_loc = dill.load(open(os.path.join(output_path_files,"anthena_loc_london_only.dill"),"rb"))
    antennas_lsoa = get_antennas_lsoa()
    user_home_antenna = pd.read_csv(os.path.join(output_path_files, "user_top_anthena_london_only_deprivation_{0}_00_04.txt".format(month)),usecols=["user_id","top_anthena"])
    user_home_antenna.dropna()
    user_home_antenna = user_home_antenna.set_index("user_id").to_dict()["top_anthena"]
    
    antenna_ids = sorted(antenna_loc.keys())
    antenna_indices = dict(zip(antenna_ids,range(len(antenna_ids))))
    antenna_indices_r = dict(zip(range(len(antenna_ids)),antenna_ids))
    group_names = sorted(user_groups.keys())#[:2]
    group_sizes = [len(user_groups[gn]) for gn in group_names]

    print group_names
    user_count = sum([len(v) for v in user_groups.itervalues()])
    
    file_names  = os.listdir(os.path.join(output_path_files,"time_sliced"))#[:2] #["hour=all"] #["hour=00-04"] 
    
#    time_slice_rank = defaultdict(float) #time slice with best mixing of groups
#    day_rank = Counter() #day with the best mixing 
#    antenna_rank = Counter() #antenna with the best mixing of groups
    
    week_aggr = defaultdict(float)
    ts_aggr = defaultdict(float)
    time_spans = ["hour=00-04","hour=04-08","hour=08-12","hour=12-16","hour=16-20","hour=20-24"]#[2:]
    time_labels = ["00-04","04-08","08-12","12-16","16-20","20-24"]#[2:]
    antenna_group_matrix = np.zeros((len(time_spans),len(antenna_ids),len(group_names))) #7 time slice
#    antenna_group_matrix = np.zeros((2,len(antenna_ids),len(group_names))) #7 time slice
    for fn in file_names:
        dstr = fn.split(".")[0].split("_")[-1]

        datetime_date = parse(dstr.split("=")[1]) 
        weekday = datetime_date.weekday()
         
        user_antennas = dill.load(open(os.path.join(output_path_files,"time_sliced",fn),"rb"))
        for i, gn in enumerate(group_names):
            user_ids = user_groups[gn]
            for uid in user_ids:
                home_antenna = user_home_antenna[uid]
                for ti, ts in enumerate(time_spans):
                    user_ant = user_antennas[ts]
                    antennas_count = user_ant[uid]
                    #remove home antenna
                    antennas_count.pop(home_antenna,None)
                    for aid, count in antennas_count.iteritems():
                        try:
                            antenna_idx = antenna_indices[aid] 
                            antenna_group_matrix[ti,antenna_idx, i] += count
                        except Exception as err:
                            pass
                            
    ts_ginis =  []        
    ts_top_antennas = {}
#    print antenna_group_matrix[0,:,:]
    for ts in xrange(len(time_spans)):
        antenna_group_matrix[ts,:,:]/= group_sizes
        #find not visited antennas
        
        sums = antenna_group_matrix[ts,:,:].sum(1)
        idx_nozero = np.where(sums!=0)[0]
#        print idx_nozero
#        print len(sums),len(idx_nozero),antenna_group_matrix[ts,:,:].shape
        aidx_ginis = {i:gini_coeff(antenna_group_matrix[ts,i,:]) for i in idx_nozero}

        ts_ginis.append(aidx_ginis.values())
        #m = np.mean(ginis)
        fname = "ts_{0}".format(time_labels[ts])

        

        #heatmap_gini_overall(aidx_ginis,antenna_indices_r,antennas_lsoa,idx_nozero,fname)
        heatmap_gini_extreems(aidx_ginis,antenna_indices_r,antennas_lsoa,idx_nozero,fname)
        
        
#    dill.dump(ts_top_antennas,open(os.path.join(output_path_files,"top100antennas_ts.dill"),"wb"))

    fig, ax = plt.subplots()
    ax.boxplot(ts_ginis,showmeans=True,labels=time_labels)
#
    plt.savefig(os.path.join(output_path_plots,"mixing_timespans.png"), bbox_inches="tight")
    plt.close() 

def anthena_user_ratio(user_groups):
#    user_anthenas = dill.load(open(os.path.join(output_path_files,"user_anthenas_august_weekend_night.dill"),"rb"))
    
    antenna_loc = dill.load(open(os.path.join(output_path_files,"anthena_loc_london_only.dill"),"rb"))
    user_home_antenna = pd.read_csv(os.path.join(output_path_files, "user_top_anthena_london_only_deprivation_{0}_00_04.txt".format(month)),usecols=["user_id","top_anthena"])
    user_home_antenna.dropna()
    user_home_antenna = user_home_antenna.set_index("user_id").to_dict()["top_anthena"]

    antenna_ids = sorted(antenna_loc.keys())
    antenna_indices = dict(zip(antenna_ids,range(len(antenna_ids))))
    group_names = sorted(user_groups.keys())
    group_sizes = [len(user_groups[gn]) for gn in group_names]

    print group_names
    user_count = sum([len(v) for v in user_groups.itervalues()])
    
    file_names  = os.listdir(os.path.join(output_path_files,"time_sliced")) #["hour=all"] #["hour=00-04"] 
    
#    time_slice_rank = defaultdict(float) #time slice with best mixing of groups
#    day_rank = Counter() #day with the best mixing 
#    antenna_rank = Counter() #antenna with the best mixing of groups
    
    week_aggr = defaultdict(float)
    ts_aggr = defaultdict(float)

    for fn in file_names:
        dstr = fn.split(".")[0].split("_")[-1]

        datetime_date = parse(dstr.split("=")[1]) 
        weekday = datetime_date.weekday()
         
        user_antennas = dill.load(open(os.path.join(output_path_files,"time_sliced",fn),"rb"))
        antenna_group_matrix = np.zeros((len(antenna_ids),len(group_names)))
        for i, gn in enumerate(group_names):
            user_ids = user_groups[gn]
            for uid in user_ids:
                home_antenna = user_home_antenna[uid]
                for time_slice, user_ant in user_antennas.iteritems():
                    antennas_count = user_ant[uid]
                    #remove home anthena
                    antennas_count.pop(home_antenna,None)
        #            anthenas_count = {k:v for k,v in anthenas_count.iteritems() if v>5}
                    for aid, count in antennas_count.iteritems():
                        try:
                            antenna_idx = antenna_indices[aid] 
        #                    anthena_group_count[anthena_idx][gn] +=count
                            antenna_group_matrix[antenna_idx, i] += count
                        except Exception as err:
        #                    print err
                            pass
            #normalize by group sizes
            
            anthenna_group_matrix/= group_sizes 
            ginis = np.apply_along_axis(gini_coeff,1,antenna_group_matrix)
            

#    row_sum = np.sum(anthena_group_matrix,1)
#    anthena_group_matrix = anthena_group_matrix[row_sum!=0]
#    anthena_group_matrix/= np.sum(anthena_group_matrix,1)[:,np.newaxis]
#    print anthena_group_matrix.shape
#    ind = np.argsort(-anthena_group_matrix)
#    temp = np.where(ind==0)[1]
#    anthena_group_matrix = anthena_group_matrix[np.argsort(temp),:]
    
#    plot_cdf(ginis,"antennas_gini")
    #sort columns 
    #mat[np.arange(np.shape(mat)[0])[:,np.newaxis],np.argsort(-mat)]

#    dill.dump(anthena_group_matrix,open(os.path.join(output_path_files,"anthena_group_matrix.dill"),"wb"))
#    print "plotting"
#    plt.figure()
#    fig, ax = plt.subplots()#
#    ax.boxplot(ginis,showmeans=True)
#
#    ax.matshow(anthena_group_matrix,cmap=plt.get_cmap("Blues"))
#    i = 0
#    for gs in group_sizes:
#        ax.axhline(y=i+gs)
#        ax.axvline(x=i+gs)
#        i+=gs
#    plt.savefig(os.path.join(output_path_plots,"anthena_group_matrix_WE.pdf"), bbox_inches="tight")
#    plt.savefig(os.path.join(output_path_plots,"antenna_group_matrix_gini_box.pdf"), bbox_inches="tight")
#    plt.close() 
 def gini_coeff(arr):
    array = arr.flatten() 
    array += 0.0000001
    array = np.sort(array)
    index = np.arange(1,array.shape[0]+1)
    n = array.shape[0]
    return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))


if __name__ == "__main__":
    users = {}
    users = get_users_group()
    ts_mixing(users)
#    daily_mixing(users)
#    similarity_matrix, group_sizes, group_names = users_trajectory_similariy(users)
#    anthena_user_ratio(users)
#    compute_avg_similarity(similarity_matrix, group_sizes, group_names)
#    users_trajectory_to_file(users)
#    gyration_bbox_compare(rich_users,poor_users)
#    user_anthena_diversity(rich_users, poor_users)
#    users_neighbourhood_diversity(rich_users, poor_users)
#    user_anthena_diversity(rich_users, poor_users)
