import os
import sys
#from dateutil.parser import parse
import datetime
from collections import defaultdict,Counter
import dill
import operator

import pandas as pd
import numpy as np

from util import output_path_files, output_path_plots,connect_mongodb, reverse_geo_mongodb,mobility_path,decompress_path

output_path_plots = os.path.join(output_path_plots,"mobility")
output_path_files = os.path.join(output_path_files,"mobility")

antenna_hash = {}

def get_active_antennas():
    #antenna_loc = dill.load(open(os.path.join(output_path_files,"antenna_loc.dill"),"rb"))
    date_dirs = [date_dir for date_dir in os.listdir(mobility_path) if date_dir.startswith("dt")]

    test_counter = 0
    for date_dir in date_dirs:
        if date_dir.endswith("247"):
            continue
        print date_dir
        path = os.path.join(mobility_path, date_dir)
        lzop_files = os.listdir(path)
        lzop_files = [i for i in lzop_files if ".lzo" in i]
        working_file_name = os.path.join(decompress_path, 'tmp_work_file')
        for lzop_f in lzop_files:
            #test_counter += 1
            print 'current size of antenna set: ' + str(len(antenna_hash))
            print 'lzop -d ' + os.path.join(path, lzop_f) + ' -o ' + working_file_name 
            #'''
            os.system('lzop -d ' + os.path.join(path,lzop_f) + ' -o ' + working_file_name)

            f = open(working_file_name, 'r')
            for line in f:
                line = line.strip()
                a = line.split("\t")
                a = filter(None,a) #remove empty strings from the list
                if len(a) == 0:
                    #print line
                    continue
                user_id = a[0]
                
                antenna_info = a[1:]
        #        print antenna_info
                if len(antenna_info)%3:
                    #there are extra columns that needs to be removed
                    antenna_info = antenna_info[:-len(antenna_info)%3]

                for i in xrange(0,len(antenna_info),3):#data fortmat: timestamp, easting, northing 
                    easting = antenna_info[i+1]
                    northing = antenna_info[i+2]

                    ID = str(easting) + "-" + str(northing)
                    #print ID
                    if ID in antenna_hash:
                        continue
                    else:
                        antenna_hash[ID] = True
                    #print antenna_info[i+1]a
            f.close()
            os.system('rm ' + working_file_name)
            if test_counter > 10:
                break

    f =  open(os.path.join(output_path_files, "active_antennas.txt"),"wb")

    for ID in antenna_hash:
        f.write("{0}\n".format(ID))
    f.close()
        #'''

if __name__ == '__main__':
    get_active_antennas()
