import os
import sys
from dateutil.parser import parse
import datetime
from collections import defaultdict,Counter
import dill
import operator
import random
import itertools

import pygeoj
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pylab as plt
from scipy import stats

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

def heatmap_gini_extreems(antennas_gini, antennas_indices,antenna_lsoa_mapping,valid_aids,fname):
    lsoa_to_coordinates = load_lsoa_polygons()
    lsoa_antennas_values = defaultdict(list)
    for aidx in valid_aids:
        aid =  antennas_indices[aidx]
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

def group_pairwise_corr(antennas_group_matrix, time_labels,number_of_groups=10):
    for t in xrange(len(time_labels)):
        group_matrix = antennas_group_matrix[t,:,:]
        print time_labels[t]
        m = np.zeros((number_of_groups,number_of_groups))
        for i,j in itertools.combinations(range(number_of_groups),2):
            a = group_matrix[:,i]
            b  = group_matrix[:,j]
            v = stats.pearsonr(a,b)[0]
            m[i,j] = v
            print "({0},{1}): {2}".format(i,j,v)
        
        fig = plt.figure() 
        ax = plt.subplot()
        cax = ax.imshow(m, interpolation='nearest', cmap=plt.cm.Blues)
        fig.colorbar(cax)
        plt.savefig(os.path.join(output_path_plots,"gp_antennas_corr","{0}.png".format(time_labels[t])))
        plt.close()

def comp_prob_std(vec):
#    print vec
    vec /= sum(vec)
    mean = sum(np.arange(len(vec)) * vec)
    var = sum( (np.arange(len(vec))**2)*vec) - (mean**2)
    std = np.sqrt(var)
#    print std
#    print '-------'
    return std

def heatmap_std(antennas_matrix,antennas_indices,antenna_lsoa_mapping, valid_aids,fname):
    lsoa_to_coordinates = load_lsoa_polygons()
    lsoa_antennas_values = defaultdict(list)
#    print antennas_indices
    for aidx in valid_aids:
        aid = antennas_indices[aidx]
        vec = antennas_matrix[aidx,:]
        try:
            lsoa_id = antenna_lsoa_mapping[aid]
            lsoa_antennas_values[lsoa_id].append(comp_prob_std(vec))
        except Exception as err:
            print err

    lsoa_values = {lsoa_id:np.mean(values) for lsoa_id,values in lsoa_antennas_values.iteritems()}
    fname = "{0}_std".format(fname)
    plot_bokeh_intensity_map(lsoa_to_coordinates, lsoa_values, fname)

def skewness(antennas_gini,antennas_matrix,time_labels,number_of_groups=10):
    results = []
    labels = []
    for t in xrange(len(time_labels)):
        tname = time_labels[t]
        labels.append(tname)
        top_antennas,middle_antennas, bottom_antennas = group_antennas(antennas_gini[t])
        top_antennas_id = [i[0] for i in top_antennas.iteritems()]
        group_matrix = antennas_matrix[t,:,:]
        top_antennas_record = group_matrix[top_antennas_id,:]
        r = stats.skew(top_antennas_record,1)
        results.append(r)

    fig = plt.figure() 
    ax = plt.subplot()
    ax.boxplot(results,labels = labels,showmeans=True)
    plt.savefig(os.path.join(output_path_plots,"top_antennas_skewness.png"))
    plt.close()



def group_antennas(antennas_gini):
    """ 
    groups antennas to top and bottom 10% and the rest
    """
    valid_aids = antennas_gini.keys()
    values = np.array(sorted(antennas_gini.values()))
    sorted_antennas = sorted(antennas_gini.items(), key=operator.itemgetter(1),reverse=True)
    l = int(len(values)*0.1) #10%
    
    top_boundry = np.where(values<values[l])[0][-1]
    bottom_boundry = np.where(values>=values[-l])[0][0]
    
    top_antennas = [i for i in sorted_antennas[top_boundry:]]
    bottom_antennas = [i for i in sorted_antennas[:bottom_boundry]]
    middle_antennas = [i for i in sorted_antennas[top_boundry+1:bottom_boundry]]
    return dict(top_antennas),dict(middle_antennas),dict(bottom_antennas)


def temporal_rank_changes_by_group(antennas_ginis,fname):
    top_ginis = {}
    bottom_ginis = {}
    middle_ginis = {}
    for tname, aids_ginis in antennas_ginis.iteritems():
        top_antennas,middle_antennas, bottom_antennas = group_antennas(aids_ginis)
        top_ginis[tname] = top_antennas
        bottom_ginis[tname] = bottom_antennas
        middle_ginis[tname] = middle_antennas
    temporal_rank_changes(top_ginis,"top_antennas_{0}".format(fname))
    temporal_rank_changes(bottom_ginis,"bottom_antennas_{0}".format(fname))
    temporal_rank_changes(middle_ginis,"middle_antennas_{0}".format(fname))

def temporal_rank_changes(antennas_ginis,fname):
    ts_ids = [set(ids) for ids in antennas_ginis.values()]
    fixed_aids = ts_ids[0]
    for i in ts_ids[1:]:
        fixed_aids.intersection(i)
    
    print "fixed antennas", len(fixed_aids)
    idx_aids = dict(zip(fixed_aids,range(len(fixed_aids))))

    t = np.zeros((len(fixed_aids),len(antennas_ginis)))
    for i in xrange(len(antennas_ginis)):
        temp = sorted(antennas_ginis[i].items(),key=operator.itemgetter(1),reverse=True)
        for j in xrange(len(temp)):
            try:
                idx = idx_aids[temp[j][0]]
                t[idx,i] = j#antennas_ginis[i][1]
            except Exception as err:
                pass
                #the antennas is not in the fixed_ids list
    stds = np.std(t,1)
    
    fig, ax = plt.subplots()
    ax.boxplot(stds,showmeans=True)

    plt.savefig(os.path.join(output_path_plots,"stds_gini_{0}.png".format(fname)), bbox_inches="tight")
    plt.close() 
    
#    ranges = np.ptp(t,1)
#    fig, ax = plt.subplots()
#    ax.boxplot(ranges,showmeans=True)
#
#    plt.savefig(os.path.join(output_path_plots,"ptp_gini_{0}.png".format(fname)), bbox_inches="tight")
#    plt.close() 


if __name__ == "__main__":
    top100antennas = dill.load(weekday_top_antennas,open(os.path.join(output_file_path,"top100antennas_ts.dill"),"rb")) 
    antenna_ids = sorted(antenna_loc.keys())
    antenna_indices = dict(zip(range(len(antenna_ids),antenna_ids)))

#    for t, aids in top100antennas:
#        with open("","wb")
