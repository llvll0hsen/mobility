import os
import sys
from collections import defaultdict, Counter
from dateutil.parser import parse

import dill

from util import *


output_path_files = os.path.join(output_path_files,"mobility")

if __name__ == '__main__':
    user_groups = dill.load(open(os.path.join(output_path_files,"user_groups_dict.dill"),"rb"))
    antenna_lsoa = dill.load(open(os.path.join(output_path_files,"antenna_lsoa_london_only.dill"),"rb"))
    time_slice = "hour=all"
    month = "august"
    mobility_path = "data/mining_mobility" 
    path = os.path.join(mobility_path, month)
    months_folders = os.listdir(path)

    user_active_days = defaultdict(int) #{user id->number of active days}
    user_bbdiagonal = defaultdict(list) #{user id->list of bbdiagonal from different days}
    user_totDuration = defaultdict(list) # {user id > list of time spent on different antenna over the whole time span}
    
    antenna_timespent = defaultdict(list) # {antenna id: [time spent at the antenna by any user]}
    group_antenna_duration = defaultdict(lambda:defaultdict(list)) #{group id:{antenna id: [list of time spent by member of the group on the antenna]}}
    group_gyration = defaultdict(lambda:defaultdict(list)) #{group id:{time label:[gyration of members of the group in a specific time span]}} 

    f_user_antenna =  open(os.path.join(output_path_files, "user_antenna_{0}_all.txt".format(month)),"wb")
    f_user_antenna.writelines("user_id,anthena_id,duration,date,weekday,time_slice")

    for d in months_folders:
        user_antenna = defaultdict(lambda: defaultdict(Counter))
        print d
        datetime_date = parse(d.split("=")[1]) 
        date_str = datetime_date.strftime("%d-%m-%y")
        weekday = datetime_date.weekday()

        path2 = os.path.join(path, d)
        time_sliced_hours = os.listdir(path2)
        for ts in time_sliced_hours:
            print ts
            try:
                files = os.listdir(os.path.join(path2,ts))
            except Exception as err:
                print err
                continue
            idx = files.index("HEADER-r-00000")
            del files[idx]
            files = [i for i in files if 'lzo' not in i]
            for f_name in files:
                print f_name
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
                    user_bbdiagonal[user_id].append(float(a[2])) 
                    user_totDuration[user_id].append(float(a[3]))
                    
                    antenna_info = a[5:]
                    if len(antenna_info)%2:
                        #sometimes there is an extra column that needs to be removed
                        antenna_info = antenna_info[:-1]
                    for i in xrange(0,len(antenna_info),2):
                        antenna_id = antenna_info[i]
                        if antenna_id in antenna_lsoa:
                            duration = antenna_info[i+1]
                            try:
                                duration = float(duration)
                                group_antenna_duration[group_id][antenna_id].append(duration) 
                            except Exception as err:
#                                print duration
#                                print line
                                print a

                        antenna_timespent[antenna_id].append(duration)
                        user_antenna[ts][user_id].update([antenna_id])
                        f_user_antenna.writelines("\n{0},{1},{2},{3},{4},{5}".format(user_id,antenna_id,duration,date_str,weekday,ts))
        
        dill.dump(user_antenna, open(os.path.join(output_path_files,"time_sliced","time_user_antennas_{0}_london_only.dill".format(d)),"wb"))
        dill.dump(group_gyration, open(os.path.join(output_path_files,"time_sliced","group_gyration_{0}_london_only.dill".format(d)),"wb"))
        dill.dump(group_antenna_duration, open(os.path.join(output_path_files,"time_sliced","group_antennas_duration_london_only_{0}.dill".format(d)),"wb"))

    
    f_user_antenna.close()
    dill.dump(user_totDuration,open(os.path.join(output_path_files,"user_totDuration.dill"),"wb"))
    dill.dump(group_gyration, open(os.path.join(output_path_files,"group_gyration.dill"),"wb"))
    dill.dump(user_bbdiagonal, open(os.path.join(output_path_files,"bbdiagonal.dill"),"wb"))
    dill.dump(user_active_days, open(os.path.join(output_path_files,"user_active_days.dill"),"wb"))
    dill.dump(antenna_timespent, open(os.path.join(output_path_files,"timespent_at_antennas.dill"),"wb"))




