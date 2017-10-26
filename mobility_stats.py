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

from util import output_path_files, output_path_plots,connect_monogdb, reverse_geo_mongodb,mobility_path

output_path_plots = os.path.join(output_path_plots,"mobility")
output_path_files = os.path.join(output_path_files,"mobility")


def home_location():
    month = "august"
    f =  open(os.path.join(output_path_files, "user_anthena_{0}_00_04.txt".format(month)),"rb")
    df = pd.read_csv(f,delimiter=",")
    df = df[(df.time_slice=="hour=00-04") & (~df.weekday.isin([5,6]))]

    temp = df.groupby(["user_id","anthena_id"]).size()
    temp = temp.unstack().idxmax(axis=1) 
    temp.to_csv(os.path.join(output_path_files, "home_locations_{0}.txt".format(month)),sep=",",index=False)

def find_home_location():
    user_anthena_duration = defaultdict(lambda :defaultdict(float))

    anthena_loc = dill.load(open(os.path.join(output_path_files,"anthena_loc.dill"),"rb"))
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
                    
                    anthena_info = a[5:]
                    if len(anthena_info)%2:
                        #there is an extra column that needs to be removed
                        anthena_info = anthena_info[:-1]
                    for i in xrange(0,len(anthena_info),2):
                        anthena_id = anthena_info[i]
                        duration = float(anthena_info[i+1])
                        user_anthena_duration[user_id][anthena_id]+=duration
        #find top anthena:
        
    f =  open(os.path.join(output_path_files, "user_top_anthena_{0}_00_04.txt".format(month)),"wb")
    f.writelines("user_id, top_anthena, lon,lat")
    f2 =  open(os.path.join(output_path_files, "missing_anthenas.txt".format(month)),"wb")
    
    user_count = 0.
    missing_counts = 0
    for user, anthena_info in user_anthena_duration.iteritems():
        user_count+=1.
        top_anthena = sorted(anthena_info.items(),key=operator.itemgetter(1),reverse=True)[0]
        try:
            print "found anthena"
            lon,lat = anthena_loc[top_anthena[0]]
        except Exception as err:
            print "missing anthena", top_anthena
            f2.writelines("{0}\n".format(top_anthena[0]))
            missing_counts+=1

        f.writelines("\n{0},{1},{2},{3}".format(user_id,top_anthena[0],lon,lat))
    f.close()
    f2.close()


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
    
    find_home_location()
    sys.exit()
    time_slice = "hour=all"
    month = "august"
    
    path = os.path.join(mobility_path, month)
    months_folders = os.listdir(path)
    user_active_days = defaultdict(int)
    user_bbdiagonal = defaultdict(list)
    user_gyration = defaultdict(list)
    user_totDuration = defaultdict(list)
    anthena_timespent =  defaultdict(list) 
#    user_anthena = defaultdict(list)
    
    print path
    f_user_anthena =  open(os.path.join(output_path_files, "user_anthena_{0}_00_04.txt".format(month)),"wb")
    f_user_anthena.writelines("user_id,anthena_id,duration,date,weekday,time_slice")

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
                    
                    user_active_days[user_id]+=1
                    user_gyration[user_id].append(float(a[1]))
                    user_bbdiagonal[user_id].append(float(a[2]))
                    user_totDuration[user_id].append(float(a[3]))
                    
                    anthena_info = a[5:]
                    if len(anthena_info)%2:
                        #there is an extra column that needs to be removed
                        anthena_info = anthena_info[:-1]
                    for i in xrange(0,len(anthena_info),2):
                        anthena_id = anthena_info[i]
                        duration = anthena_info[i+1]
                        try:
                            duration = float(duration)
                        except Exception as err:
                            print duration
                            print line
                            print a
                            sys.exit()

                        anthena_timespent[anthena_id].append(duration)
                        #user_anthena[user_id].append((anthena_info[i],anthena_info[i+1]))
                        f_user_anthena.writelines("\n{0},{1},{2},{3},{4},{5}".format(user_id,anthena_id,duration,date_str,weekday,ts))
        


#    plot_boxplot(user_gyration,"gyration")
#    plot_boxplot(user_bbdiagonal,"bbdiagonal")
#    plot_boxplot(user_totDuration,"totDuration")
#    plot_boxplot(anthena_timespent,"anthenaPopularity")
    f_user_anthena.close()
    
#    dill.dump(user_active_days, open("test.dill","wb"))





