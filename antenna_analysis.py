import os
import sys
from dateutil.parser import parse
import datetime
from collections import defaultdict,Counter
import dill
import operator
import random

import pygeoj
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pylab as plt
from scipy.sparse import lil_matrix
from sklearn.metrics.pairwise import cosine_similarity

from plot_bookeh import plot_bokeh_intensity_map
from util import output_path_files, output_path_plots,connect_monogdb, reverse_geo_mongodb,mobility_path,census_data_fpath

output_path_plots = os.path.join(output_path_plots,"mobility")
output_path_files = os.path.join(output_path_files,"mobility")
month ="august"

def load_lsoa_polygons():
    polygon_data = pygeoj.load(filepath="lsoa_org.geojson")
    spatial_entity_to_coordinates = {}
    for feature in polygon_data:
        coords = feature.geometry.coordinates
        coords = coords[0]
        
        lsoa_id = feature.properties['lsoa01cd']
#        print lsoa_id
        try:
            xs = [i for i,j in coords]
            ys = [j for i,j in coords]
        except Exception as err:
            coords = coords[0]
            xs = [i for i,j in coords]
            ys = [j for i,j in coords]

        spatial_entity_to_coordinates[lsoa_id] = [xs,ys]
    return spatial_entity_to_coordinates

def heatmap_gini_overall(antennas_gini, antennas_indices,antenna_lsoa_mapping,valid_aids,fname):
    lsoa_to_coordinates = load_lsoa_polygons()
    lsoa_antennas_values = defaultdict(list)
#    print antennas_indices
    for aidx in valid_aids:
        aid = antennas_indices[aidx]
        try:
            lsoa_id = antenna_lsoa_mapping[aid]
            lsoa_antennas_values[lsoa_id].append(antennas_gini[aidx])
        except Exception as err:
            print err

    lsoa_values = {lsoa_id:np.mean(values) for lsoa_id,values in lsoa_antennas_values.iteritems()}
    plot_bokeh_intensity_map(lsoa_to_coordinates, lsoa_values, fname)

def heatmap_gini_extreems(antennas_gini, antennas_indices,antenna_lsoa_mapping,fname):
    lsoa_to_coordinates = load_lsoa_polygons()
    lsoa_antennas_values = defaultdict(list)
    for aid, aidx in antennas_indices.iteritems():
        try:
            lsoa_id = antenna_lsoa_mapping[aid]
            lsoa_antennas_values[lsoa_id].append(antennas_gini[aidx])
        except Exception as err:
            err

    lsoa_values = {lsoa_id:np.mean(values) for lsoa_id,values in lsoa_antennas_values.iteritems()}
    values = sorted(lsoa_values.values())
    l = int(len(values)*0.1)
    top_val = values[l]
    bottom_val = values[-l]
    lsoa_values = {lsoa_id:val for lsoa_id,val in lsoa_values.iteritems() if val<=top_val or val>=bottom_val} #smallest gini coeff
    fname = "extreems_{0}".format(fname)
    plot_bokeh_intensity_map(lsoa_to_coordinates, lsoa_values, fname)

if __name__ == "__main__":
    top100antennas = dill.load(weekday_top_antennas,open(os.path.join(output_file_path,"top100antennas_ts.dill"),"rb")) 
    antenna_ids = sorted(antenna_loc.keys())
    antenna_indices = dict(zip(range(len(antenna_ids),antenna_ids)))

#    for t, aids in top100antennas:
#        with open("","wb")
