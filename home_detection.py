import os
import sys
from dateutil.parser import parse
import datetime
from collections import defaultdict,Counter
import dill
import operator

import pandas as pd
import numpy as np

from util import output_path_files, output_path_plots,connect_mongodb, reverse_geo_mongodb,mobility_path

output_path_plots = os.path.join(output_path_plots,"mobility")
output_path_files = os.path.join(output_path_files,"mobility")
def find_home_antenna():
    user_antenna_duration = defaultdict(lambda :defaultdict(float))

    antenna_loc = dill.load(open(os.path.join(output_path_files,"antenna_loc.dill"),"rb"))
    month = "august" #only use august 
    path = os.path.join(mobility_path, month)
    months_folders = os.listdir(path)
    for d in months_folders:
        print d
        datetime_date = parse(d.split("=")[1]) 
        date_str = datetime_date.strftime("%d-%m-%y")
        weekday = datetime_date.weekday()
        if weekday not in [4,5]:#skip weekends - friday and saturday
            path2 = os.path.join(path, d)
            time_sliced_hours =["hour=00-04"] #only consider midnigth to 4am
            for ts in time_sliced_hours:
                print ts
                try: # for some days there is no data for 00-04
                    files = os.listdir(os.path.join(path2,ts))
                except Exception as err:
                    print err
                    continue
                idx = files.index("HEADER-r-00000")
                del files[idx]
                #exclude .izo files
                files = [i for i in files if ".lzo" not in i]
                for f_name in files:
                    print f_name
                    fpath = os.path.join(path2, ts,f_name)
                    f = open(fpath,"rb")
                    for line in f:
                        line = line.strip()
                        a = line.split("\t")
                        a = filter(None,a) #remove empty strings from the list
                        user_id = a[0]
                        
                        antenna_info = a[5:]
#                        print antenna_info
                        if len(antenna_info)%2:
                            #there is an extra column that needs to be removed
                            antenna_info = antenna_info[:-1]
                        for i in xrange(0,len(antenna_info),2):#data fortmat: antenna id, duration 
                            antenna_id = antenna_info[i]
                            #print antenna_info[i+1]
                            try:
                                duration = float(antenna_info[i+1])
                                user_antenna_duration[user_id][antenna_id]+=duration
                            except Exception as err:
                                print err
                                print antenna_info
                                sys.exit()

    f =  open(os.path.join(output_path_files, "user_top_antenna_{0}_00_04.txt".format(month)),"wb")
    f.write("user_id, top_antenna, lon, lat\n")

    f2 =  open(os.path.join(output_path_files, "missing_antennas.txt".format(month)),"wb")
    
    user_count = 0.
    missing_counts = 0
    for user, antenna_info in user_antenna_duration.iteritems():
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
    f_out.writelines("user_id,top_antenna,lon,lat,lsoa")
    f.next()
    r = dill.load(open(os.path.join(output_path_files,"antenna_loc_london_only.dill"),"rb"))
    r2 = dill.load(open(os.path.join(output_path_files,"antenna_lsoa_london_only.dill"),"rb"))
    for line in f:
        user_id, top_antenna, lon, lat = line.strip().split(",")
        if top_antenna in r:
            f_out.writelines("\n{0},{1},{2},{3},{4}".format(user_id,top_antenna,lon,lat,r2[top_antenna]))
    f.close()
    f_out.close()

if __name__ == '__main__':
    find_home_antenna()
    home_location_london_only()
