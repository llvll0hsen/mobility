import os
import sys
from dateutil.parser import parse
import datetime
from collections import defaultdict,Counter
import dill
import operator

import pandas as pd
import numpy as np

from util import output_path_files, output_path_plots,connect_mongodb, reverse_geo_mongodb,mobility_path,decompress_path

output_path_plots = os.path.join(output_path_plots,"mobility")

def find_home_antenna(month):
    user_home_antenna = {}

    path = os.path.join(decompress_path, month, 'hour00_04')
    files = os.listdir(path) 
    files = [i for i in files if ".lzo" in i]
    working_file_name = os.path.join(decompress_path, 'tmp_work_file')

    test_counter = 0
    for f_name in files:
        test_counter += 1
        print 'lzop -d ' + os.path.join(path, f_name) + ' -o ' + working_file_name 
        os.system('lzop -d ' + os.path.join(path,f_name) + ' -o ' + working_file_name)

        f = open(working_file_name, 'r')
        for line in f:
            user_buffer = defaultdict(float)
            line = line.strip()
            a = line.split("\t")
            a = filter(None,a) #remove empty strings from the list
            user_id = a[0]
            
            antenna_info = a[5:]
#            print antenna_info
            if len(antenna_info)%2:
                #there is an extra column that needs to be removed
                antenna_info = antenna_info[:-1]
            for i in xrange(0,len(antenna_info),2):#data fortmat: antenna id, duration 
                antenna_id = antenna_info[i]
                #print antenna_info[i+1]
                try:
                    duration = float(antenna_info[i+1])
                    #user_antenna_duration[user_id][antenna_id]+=duration
                    user_buffer[antenna_id]+=duration
                except Exception as err:
                    print err
                    print antenna_info
                    sys.exit()
            top_antenna = sorted(user_buffer.items(),key=operator.itemgetter(1),reverse=True)[0]
            user_home_antenna[user_id] = top_antenna[0]
            user_buffer = None

        f.close()
        os.system('rm ' + working_file_name)
        if test_counter > 10:
            break

    f =  open(os.path.join(output_path_files, "{0}_user_home_antenna.txt".format(month)),"wb")
    f.write("user_id, home_antenna\n")
    
    user_count = 0
    missing_counts = 0
    for user in user_home_antenna:
        user_count+=1
        f.write("{0},{1}\n".format(user,user_home_antenna[user]))

    f.close()
    print 'total # of users: ' + str(user_count)

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
    if len(sys.argv) < 2:
        print "specify the month to process"
    else:
        find_home_antenna(sys.argv[1])
        #home_location_london_only()
