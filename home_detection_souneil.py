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
    path = os.path.join(mobility_path, 'aggregates6m', 'all')
    lzop_files = os.listdir(path)
    lzop_files = [i for i in lzop_files if ".lzo" in i]
    working_file_name = os.path.join(mobility_path, 'tmp_work_file')
    for lzop_f in lzop_files:
        #print 'lzop -d ' + os.path.join(path, lzop_f) + ' -o ' + working_file_name 
        #'''
        os.system('lzop -d ' + os.path.join(path,lzop_f) + ' -o ' + working_file_name)

        f = open(working_file_name, 'r')
        for line in f:
            line = line.strip()
            a = line.split("\t")
            a = filter(None,a) #remove empty strings from the list
            user_id = a[0]
            
            antenna_info = a[1:]
    #        print antenna_info
            if len(antenna_info)%3:
                #there are extra columns that needs to be removed
                antenna_info = antenna_info[:-len(antenna_info)%3]

            for i in xrange(0,len(antenna_info),3):#data fortmat: antenna id, freq, duration 
                antenna_id = antenna_info[i]
                #print antenna_info[i+1]
                try:
                    duration = float(antenna_info[i+2])
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
        os.system('rm ' + working_file_name)
        #'''
    sys.exit()

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
