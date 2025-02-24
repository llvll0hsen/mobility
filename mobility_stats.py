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

from util import output_path_files, output_path_plots,connect_mongodb, reverse_geo_mongodb,mobility_path

output_path_plots = os.path.join(output_path_plots,"mobility")
output_path_files = os.path.join(output_path_files,"mobility")

def get_antenna_lsoa():
    antenna_loc = dill.load(open(os.path.join(output_path_files,"antenna_loc_london_only_new.dill"),"rb"))
    collection,client = connect_mongodb()
    antenna_lsoa = {}
    missing_antennas = []
    for antenna_id, loc in antenna_loc.iteritems():
#        print loc
#        print loc,antenna_id
        lon,lat =  loc
#        print loc
        try:
            ne = reverse_geo_mongodb(float(lat.strip()),float(lon.strip()),collection)
            print ne
            antenna_lsoa[antenna_id] = ne[0]
        except Exception as err:
            print err
#            print loc
            pass
    
    dill.dump(antenna_lsoa, open(os.path.join(output_path_files,"antenna_lsoa_london_only.dill"),"wb"))
    client.close()
def find_missing_aids():
    aid_lsoa = dill.load(open(os.path.join(output_path_files,"antenna_lsoa_london_only_new.dill"),"rb"))
    print len(list(set(aid_lsoa.values())))
    antenna_loc = dill.load(open(os.path.join(output_path_files,"antenna_loc_london_only_new.dill"),"rb"))
    missing_aids = list(set(antenna_loc.keys()) - set(aid_lsoa.keys()))
    print len(missing_aids) 
    print antenna_loc[missing_aids[0]]
    print antenna_loc[missing_aids[1]]
    print antenna_loc[missing_aids[2]]
    print len(missing_aids)
    print "antennas london lsoa", len(set(aid_lsoa.values()))
    print "percentage of missing aids: {0}".format(float(len(missing_aids))/len(antenna_loc))
    
    print missing_aids
    dill.dump(list(missing_aids),open(os.path.join(output_path_files,"missing_aids_lsoa.dill"),"wb"))

def antenna_neighborhood():
    anthena_loc = dill.load(open(os.path.join(output_path_files,"anthena_loc_london_only.dill"),"rb"))
    collection,client = connect_monogdb()
    anthena_ne = {}
    for anthena_id, loc in anthena_loc.iteritems():
#        print loc
        lon,lat =  loc
        try:
            ne = reverse_geo_mongodb(float(lat.strip()),float(lon.strip()),collection)
            anthena_ne[anthena_id] = ne[0]
        except Exception as err:
            print err
            print loc
            pass

    dill.dump(anthena_ne, open(os.path.join(output_path_files,"anthena_ne_london_only.dill"),"wb"))
    client.close()


def home_location():
    month = "august"
    f =  open(os.path.join(output_path_files, "user_antenna_{0}_00_04.txt".format(month)),"rb")
    df = pd.read_csv(f,delimiter=",")
    df = df[(df.time_slice=="hour=00-04") & (~df.weekday.isin([5,6]))]

    temp = df.groupby(["user_id","anthena_id"]).size()
    temp = temp.unstack().idxmax(axis=1) 
    temp.to_csv(os.path.join(output_path_files, "home_locations_{0}.txt".format(month)),sep=",",index=False)

def find_home_location():
    user_antenna_duration = defaultdict(lambda :defaultdict(float))

    antenna_loc = dill.load(open(os.path.join(output_path_files,"antenna_loc_new.dill"),"rb"))
    month = "august"
    path = os.path.join(mobility_path, month)
    months_folders = os.listdir(path)
    for d in months_folders:
        print d
        datetime_date = parse(d.split("=")[1]) 
        date_str = datetime_date.strftime("%d-%m-%y")
        weekday = datetime_date.weekday()

        path2 = os.path.join(path, d)
        time_sliced_hours =["hour=00-04"] #os.listdir(path2)
        for ts in time_sliced_hours:
            print ts
            try:
                files = os.listdir(os.path.join(path2,ts))
            except Exception as err:
                print err
                continue
            idx = files.index("HEADER-r-00000")
            del files[idx]
            for f_name in files:
                fpath = os.path.join(path2, ts,f_name)
                f = open(fpath,"rb")
                for line in f:
                    line = line.strip()
                    a = line.split("\t")
                    a = filter(None,a) #remove empty strings from the list
                    user_id = a[0]
                    
                    antenna_info = a[5:]
                    if len(antenna_info)%2:
                        #there is an extra column that needs to be removed
                        antenna_info = antenna_info[:-1]
                    for i in xrange(0,len(antenna_info),2):
                        antenna_id = antenna_info[i]
                        duration = float(antenna_info[i+1])
                        user_antenna_duration[user_id][antenna_id]+=duration
        #find top anthena:
        
    f =  open(os.path.join(output_path_files, "user_top_antenna_{0}_00_04.txt".format(month)),"wb")
    f.write("user_id, top_antenna, lon, lat\n")

    f2 =  open(os.path.join(output_path_files, "missing_antennas.txt".format(month)),"wb")
    
    user_count = 0.
    missing_counts = 0
    for user, antenna_info in user_antenna_duration.iteritems():
#        print user
        user_count+=1.
        top_antenna = sorted(antenna_info.items(),key=operator.itemgetter(1),reverse=True)[0]
        try:
#            print "found anthena"
            lon,lat = antenna_loc[top_antenna[0]]
            f.write("{0},{1},{2},{3}".format(user,top_antenna[0],lon,lat))
        except Exception as err:
#            print "missing anthena", top_anthena
            f2.write("{0}\n".format(top_antenna[0]))
            missing_counts+=1

    f.close()
    f2.close()

def home_location_london_only():
    
    month = "august"
    f =  open(os.path.join(output_path_files, "user_top_antenna_{0}_00_04.txt".format(month)),"rb")
    f_out =  open(os.path.join(output_path_files, "user_top_antenna_london_only_{0}_00_04.txt".format(month)),"wb")
    f_out.writelines("user_id,top_antenna,lon,lat,neighborhood")
    f.next()
    r = dill.load(open(os.path.join(output_path_files,"antenna_loc_london_only_new.dill"),"rb"))
    r2 = dill.load(open(os.path.join(output_path_files,"antenna_lsoa_london_only_new.dill"),"rb"))
    for line in f:
#        print line
        user_id, top_antenna, lon, lat = line.strip().split(",")
#        print lon,lat
        if top_antenna in r:
#            print ne
            f_out.writelines("\n{0},{1},{2},{3},{4}".format(user_id,top_antenna,lon,lat,r2[top_antenna]))
    f.close()
    f_out.close()


def plot_cdf(data):
    hist = Counter(data.values())
    n = float(sum(hist.values()))
    normalized_count = {day: freq/n for day,freq in hist.iteritems()}
    sorted_pk = sorted(normalized_count.iteritems(), key=operator.itemgetter(0),reverse=True)
    x = [i[0] for i in sorted_pk]
    y = np.cumsum([i[1] for i in sorted_pk])
    
    plt.figure() 
    ax = plt.subplot()
    a = ax.plot(x,y ,lw=5., alpha=0.6)
    ax.tick_params(direction='out')
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False) 
    ax.get_xaxis().tick_bottom()  
    ax.get_yaxis().tick_left()
    ax.set_xlabel("d",fontsize=25)
    ax.set_ylabel("x>=d",fontsize=25)
    ax.axis('tight')
    ax.margins(0.05)
    plt.savefig(os.path.join(output_path_plots,"user_activity.png"),bbox_inches='tight')

def plot_boxplot(data,title):
    mean_vals = [np.mean(vals) for user, vals in data.iteritems()]
    plt.figure()
    ax = plt.subplot()
    ax.boxplot(mean_vals)
    ax.set_title(title)
    plt.savefig(os.path.join(output_path_plots,"{0}.png".format(title)),bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    
#    find_home_location()
#    antenna_neighborhood()
#    home_location_london_only()

#    get_antenna_lsoa()

#    find_missing_aids()
#    sys.exit()
    user_groups = dill.load(open(os.path.join(output_path_files,"user_groups_dict.dill"),"rb"))
    antenna_lsoa = dill.load(open(os.path.join(output_path_files,"antenna_lsoa_london_only_new.dill"),"rb"))
#    print user_groups.keys()[:5]
#    time_slice = "hour=all"
    month = "august"
    mobility_path = "data/mining_mobility" 
    path = os.path.join(mobility_path, month)
    months_folders = os.listdir(path)
    user_active_days = defaultdict(int)
    user_bbdiagonal = defaultdict(list)
    group_gyration = defaultdict(lambda:defaultdict(list))
    user_totDuration = defaultdict(list)
    group_antenna_duration = defaultdict(lambda:defaultdict(list))

    anthena_timespent =  defaultdict(list) 
#    user_anthenna = defaultdict(lambda: defaultdict(Counter))
    antenna_set = set()
#    print path
#    f_user_anthena =  open(os.path.join(output_path_files, "user_anthena_{0}_all.txt".format(month)),"wb")
#    f_user_anthena.writelines("user_id,anthena_id,duration,date,weekday,time_slice")

    for d in months_folders:
        user_antenna = defaultdict(lambda: defaultdict(Counter))
        print d
        datetime_date = parse(d.split("=")[1]) 
        date_str = datetime_date.strftime("%d-%m-%y")
        weekday = datetime_date.weekday()

        path2 = os.path.join(path, d)
        time_sliced_hours = os.listdir(path2) #["hour=all"] #["hour=00-04"] 
        for ts in time_sliced_hours:
            print ts
            try:
                files = os.listdir(os.path.join(path2,ts))
            except Exception as err:
                print err
                continue
            idx = files.index("HEADER-r-00000")
            del files[idx]
            for f_name in files:
                fpath = os.path.join(path2, ts,f_name)
                f = open(fpath,"rb")
                for line in f:
                    line = line.strip()
                    a = line.split("\t")
                    a = filter(None,a) #remove empty strings from the list
                    user_id = a[0]
                    user_active_days[user_id] += 1
                    try:
                        group_id  = user_groups[user_id]
                        group_gyration[group_id][ts].append(float(a[1]))
                    except Exception as err:
                        group_id = -1
#                    user_bbdiagonal[user_id].append(float(a[2]))
#                    user_totDuration[user_id].append(float(a[3]))
                    
                    anthena_info = a[5:]
                    if len(anthena_info)%2:
                        #there is an extra column that needs to be removed
                        anthena_info = anthena_info[:-1]
                    for i in xrange(0,len(anthena_info),2):
                        antenna_id = anthena_info[i]
                        if antenna_id in antenna_lsoa:
                            antenna_set.add(antenna_id)
                            duration = anthena_info[i+1]
                            try:
                                duration = float(duration)
                                group_antenna_duration[group_id][antenna_id].append(duration) 
                            except Exception as err:
                                print duration
                                print line
                                print a
#                            sys.exit()

#                        anthenna_timespent[anthena_id].append(duration)
#                        print user_id, anthena_id
                        user_antenna[ts][user_id].update([antenna_id])
#                        user_anthenna[user_id][ts].update([anthena_id])
#                        user_anthena[user_id].append((anthena_info[i],anthena_info[i+1]))
#                        f_user_anthena.writelines("\n{0},{1},{2},{3},{4},{5}".format(user_id,anthena_id,duration,date_str,weekday,ts))
        
        dill.dump(user_antenna, open(os.path.join(output_path_files,"time_sliced","time_user_antennas_{0}_london_only.dill".format(d)),"wb"))
        dill.dump(group_gyration, open(os.path.join(output_path_files,"time_sliced","group_gyration_{0}_london_only.dill".format(d)),"wb"))
        dill.dump(group_antenna_duration, open(os.path.join(output_path_files,"time_sliced","group_antennas_duration_london_only_{0}.dill".format(d)),"wb"))
    dill.dump(antenna_set, open(os.path.join(output_path_files,"antenna_set.dill"),"wb"))

#    print user_anthenna.items()
#    plot_boxplot(user_gyration,"gyration")
#    plot_boxplot(user_bbdiagonal,"bbdiagonal")
#    plot_boxplot(user_totDuration,"totDuration")
#    plot_boxplot(anthena_timespent,"anthenaPopularity")
#    f_user_anthena.close()
    
#    dill.dump(user_totDuration, open("user_gyration.dill","wb"))
#    dill.dump(user_gyration, open("user_gyration.dill","wb"))
#    dill.dump(user_bbdiagonal, open("bbdiagonal.dill","wb"))

#    dill.dump(user_active_days, open("test.dill","wb"))
#    dill.dump(user_anthenna, open(os.path.join(output_path_files,"user_antennas_august_all_timesliced.dill"),"wb"))





