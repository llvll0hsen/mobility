import os
import sys
import itertools
from collections import defaultdict

import numpy as np
import dill
from vincenty import vincenty

from util import output_path_files

def antennas_dist():
    antenna_coverage = 500
    aids_loc = dill.load(open(os.path.join(output_path_files,"mobility","antenna_loc_london_only_new.dill"),"rb"))
#    aids_loc = aids_loc.items()
    dists = dill.load(open(os.path.join(output_path_files,"mobility","antennas_dist.dill"),"rb"))
#    dists = defaultdict(dict)
#    for i,j in itertools.combinations(range(len(aids_loc)),2):
#        aid_i,loc_i = aids_loc[i]
#        aid_j,loc_j = aids_loc[j]
#        
#        lon_i,lat_i = loc_i
#        lon_j,lat_j = loc_j
#
#        d = vincenty((float(lon_i),float(lat_i)),(float(lon_j),float(lat_j)))
#        dists[aid_i][aid_j] = d
    for aid in aids_loc.iterkeys():
        dists[aid][aid] = antenna_coverage

    dill.dump(dists,open(os.path.join(output_path_files,"mobility","antennas_dist.dill"),"wb"))

if __name__ == '__main__':
    antennas_dist()
