import traceback
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
from util import output_path_files, output_path_plots,connect_mongodb, reverse_geo_mongodb,mobility_path,census_data_fpath,get_depr_factor
from dist_to_center import dist_corr
from segregation_indices import get_exposure_index, get_diss_index, get_isolation_index, get_spatial_proximity

output_path_plots = os.path.join(output_path_plots,"mobility")
output_path_files = os.path.join(output_path_files,"mobility")
month ="august"

def load_lsoa_polygons():
    polygon_data = pygeoj.load(filepath="lsoa_new_nodup.geojson")
    spatial_entity_to_coordinates = {}
    for feature in polygon_data:
        coords = feature.geometry.coordinates
        coords = coords[0]
        
        lsoa_id = feature.properties['LSOA11CD']
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

def group_pairwise_corr(group_matrix,time_label,number_of_groups=10):
    m = np.zeros((number_of_groups,number_of_groups))
    for i,j in itertools.combinations(range(number_of_groups),2):
        a = group_matrix[:,i]
        b  = group_matrix[:,j]
        v = stats.pearsonr(a,b)[0]
        m[i,j] = v
#        print "({0},{1}): {2}".format(i,j,v)
    
    fig = plt.figure() 
    ax = plt.subplot()
    cax = ax.imshow(m, interpolation='nearest', cmap=plt.cm.seismic, vmin=-0.3,vmax=0.3)
    fig.colorbar(cax)
    plt.savefig(os.path.join(output_path_plots,"gp_antennas_corr","{0}.png".format(time_label)))
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

def std_dist(stds,fname):
    fig, ax = plt.subplots()
    ax.hist(stds,10,alpha=0.7)
#    ax.set_title("{0}: {1} 10%".format(dpr_name,pname))
    ax.set_xlabel("std")

    plt.savefig(os.path.join(output_path_plots,"stds_hist","{0}.png".format(fname)), bbox_inches="tight")
    plt.close() 
def heatmap_std(antennas_matrix,antennas_indices,antenna_lsoa_mapping, valid_aids,fname):
#    antenna_depr = dill.load(open(os.path.join(output_path_files,"antenna_depr.dill"),"rb"))
    lsoa_to_coordinates = load_lsoa_polygons()
    lsoa_antennas_values = defaultdict(list)
#    print antennas_indices
    antennas_std = {}
    antennas_std2 = {}
    i = 0
    for aidx in valid_aids:
        aid = antennas_indices[aidx]
        vec = antennas_matrix[aidx,:]
        try:
#            print aid, aidx
            lsoa_id = antenna_lsoa_mapping[aid]
            temp_std = comp_prob_std(vec)
            antennas_std[aid] = temp_std
            antennas_std2[aidx] = temp_std
            lsoa_antennas_values[lsoa_id].append(temp_std)
        except Exception as err:
#            traceback.print_exc()
            pass
            i+=1

#    print "missing  aids", i
#    f = open(os.path.join(output_path_files,"dist_center_corr","corr.txt"),"ab")
#    dist_corr(antennas_std,fname) 
     
#    lssoa_values = {lsoa_id:np.mean(values) for lsoa_id,values in lsoa_antennas_values.iteritems()}
#    all_stds = lsoa_values.values()
#    std_dist(all_stds,fname)
#    subdomains = ["Children and Young People Sub-domain Decile (where 1 is most deprived 10% of LSOAs)","Adult Skills Sub-domain Decile (where 1 is most deprived 10% of LSOAs)","Geographical Barriers Sub-domain Decile (where 1 is most deprived 10% of LSOAs)","Wider Barriers Sub-domain Decile (where 1 is most deprived 10% of LSOAs)","Indoors Sub-domain Decile (where 1 is most deprived 10% of LSOAs)","Outdoors Sub-domain Decile (where 1 is most deprived 10% of LSOAs)"]
    factors = ["IMD Decile (where 1 is most deprived 10% of LSOAs)", "Income Decile (where 1 is most deprived 10% of LSOAs)","Employment Decile (where 1 is most deprived 10% of LSOAs)","Education, Skills and Training Decile (where 1 is most deprived 10% of LSOAs)","Health Deprivation and Disability Decile (where 1 is most deprived 10% of LSOAs)",
            "Crime Decile (where 1 is most deprived 10% of LSOAs)","Barriers to Housing and Services Decile (where 1 is most deprived 10% of LSOAs)",
            "Living Environment Decile (where 1 is most deprived 10% of LSOAs)"]
#    fname2 = "{0}_std_corr.txt".format(fname)
#    f = open(os.path.join(output_path_files,"dpr_corr",fname2),"wb")
#
#    for factor_name in factors:
##        print factor_name
#        lsoa_depr = get_depr_factor(factor_name)
##        std_margines_lsoa_dep_index(antennas_matrix,antennas_std, antennas_indices, antenna_lsoa_mapping, lsoa_depr, factor_name, fname)
#        deprivation_corr(lsoa_depr, lsoa_values, factor_name,f)
#    f.close()
#    std_class_corr(antennas_matrix, antennas_std2, fname)
#    exposure_index(antennas_matrix, antennas_std2, fname)
#    dissimilarity_index(antennas_matrix, antennas_std2, fname)
#    isolation_index(antennas_matrix, antennas_std2, fname)
    spatial_proximity(antennas_matrix, antennas_std2, antennas_indices, fname)

def deprivation_corr(antenna_depr, antenna_std,factor_name,f):
    print len(antenna_depr), len(antenna_std)
    antenna_depr = {k.lower():antenna_depr[k.lower()] for k in antenna_std.iterkeys()}

    std_sorted = sorted(antenna_std.items(), key=operator.itemgetter(1),reverse=True)
    #depr_rank = sorted(antenna_depr.items(), key=operator.itemgetter(1),reverse=True)
    depr_ranks = []
    std_ranks = range(len(std_sorted))
    for i,v in enumerate(std_sorted):
        depr_ranks.append( antenna_depr[v[0].lower()])

    corr = stats.pearsonr(np.array(std_ranks),np.array(depr_ranks)*-1)
    f.writelines("{0}\t{1}\n".format(factor_name,corr))
#    print fname,corr

def exposure_index(antennas_matrix, antennas_std, time_label):
    top,middle,bottom = group_antennas(antennas_std,0.1)
    top_antennas_matrix = antennas_matrix[np.array(top.keys()),:]
    bottom_antennas_matrix = antennas_matrix[np.array(bottom.keys()),:]

    fname_top = "exposure_top_{0}_10.png".format(time_label)
    fname_bottom = "exposure_bottom_{0}_10.png".format(time_label)
    fname_all = "exposure_all_{0}_10.png".format(time_label)
    B_top = get_exposure_index(top_antennas_matrix)
    B_bottom = get_exposure_index(bottom_antennas_matrix)
    valid_aids = np.array(antennas_std.keys())
    B_all = get_exposure_index(antennas_matrix[valid_aids,:])
    d = {fname_top:B_top, fname_bottom:B_bottom,fname_all:B_all}
    for fname, data in d.iteritems():
        fig = plt.figure() 
        ax = plt.subplot()
        cax = ax.imshow(data, interpolation='nearest', cmap=plt.cm.Blues)
        for (j,i),label in np.ndenumerate(data):
            ax.text(i,j,label,ha="center",va="center")
        fig.colorbar(cax)
        plt.savefig(os.path.join(output_path_plots,"exposure",fname))
        plt.close()

#        with open(os.path.join(output_path_files,'exposure',fname),'wb') as f:
#            for row in data:
#                l = "\t".join(map(str,row))
#                f.write(l+'\n')
def spatial_proximity(antennas_matrix,antennas_std,antenna_indices,time_label):
    top,middle,bottom = group_antennas(antennas_std,0.1)
#    top_antennas_matrix = antennas_matrix[np.array(top.keys()),:]
#    bottom_antennas_matrix = antennas_matrix[np.array(bottom.keys()),:]
    


    fname_top = "top_{0}_10.png".format(time_label)
    fname_bottom = "bottom_{0}_10.png".format(time_label)
    fname_all = "all_{0}_10.png".format(time_label)
    D_top = get_spatial_proximity(antennas_matrix,top.keys(),antenna_indices)
    D_bottom = get_spatial_proximity(antennas_matrix, bottom.keys(),antenna_indices)
    valid_aids = np.array(antennas_std.keys())
    D_all = get_spatial_proximity(antennas_matrix, valid_aids,antenna_indices)
    d = {fname_top:D_top, fname_bottom:D_bottom,fname_all:D_all}
    for fname, data in d.iteritems():
        print fname
        fig = plt.figure() 
        ax = plt.subplot()
        cax = ax.imshow(data, interpolation='nearest', cmap=plt.cm.Blues)
        for i,j in itertools.combinations(range(10),2):
            ax.text(j,i,data[i,j],ha="center",va="center")
        fig.colorbar(cax)
        plt.savefig(os.path.join(output_path_plots,"spatial_proximity",fname))
        plt.close()
def dissimilarity_index(antennas_matrix, antennas_std, time_label):
    top,middle,bottom = group_antennas(antennas_std,0.1)
    top_antennas_matrix = antennas_matrix[np.array(top.keys()),:]
    bottom_antennas_matrix = antennas_matrix[np.array(bottom.keys()),:]

    fname_top = "diss_top_{0}_10.png".format(time_label)
    fname_bottom = "diss_bottom_{0}_10.png".format(time_label)
    fname_all = "diss_all_{0}_10.png".format(time_label)
    D_top = get_diss_index(top_antennas_matrix)
    D_bottom = get_diss_index(bottom_antennas_matrix)
    valid_aids = np.array(antennas_std.keys())
    D_all = get_diss_index(antennas_matrix[valid_aids,:])
    d = {fname_top:D_top, fname_bottom:D_bottom,fname_all:D_all}
    for fname, data in d.iteritems():
        fig = plt.figure() 
        ax = plt.subplot()
        cax = ax.imshow(data, interpolation='nearest', cmap=plt.cm.Blues)
        for i,j in itertools.combinations(range(10),2):
            ax.text(j,i,data[i,j],ha="center",va="center")
        fig.colorbar(cax)
        plt.savefig(os.path.join(output_path_plots,"dissimilarity",fname))
        plt.close()

def isolation_index(antennas_matrix,antennas_std,time_label):
    top,middle,bottom = group_antennas(antennas_std,0.1)
    top_antennas_matrix = antennas_matrix[np.array(top.keys()),:]
    bottom_antennas_matrix = antennas_matrix[np.array(bottom.keys()),:]

    fname_top = "isolation_top_{0}_10.png".format(time_label)
    fname_bottom = "isolation_bottom_{0}_10.png".format(time_label)
    fname_all = "isolation_all_{0}_10.png".format(time_label)
    D_top = get_isolation_index(top_antennas_matrix)
    D_bottom = get_isolation_index(bottom_antennas_matrix)
    valid_aids = np.array(antennas_std.keys())
    D_all = get_isolation_index(antennas_matrix[valid_aids,:])
    d = {fname_top:D_top, fname_bottom:D_bottom,fname_all:D_all}
    x = range(len(D_top))
    for fname, data in d.iteritems():
        fig = plt.figure() 
        ax = plt.subplot()
        cax = ax.plot(x,data)
        plt.savefig(os.path.join(output_path_plots,"isolation",fname))
        plt.close()

def std_class_corr(antennas_matrix, antennas_std, time_label):
    top,middle,bottom = group_antennas(antennas_std,0.1)
#    print top.keys()
    top_antennas_matrix = antennas_matrix[np.array(top.keys()),:]
    bottom_antennas_matrix = antennas_matrix[np.array(bottom.keys()),:]

    fname_top = "top_{0}_10".format(time_label)
    fname_bottom = "bottom_{0}_10".format(time_label)
#    fname_all = "all_{0}".format(time_label)

    group_pairwise_corr(top_antennas_matrix, fname_top,number_of_groups=10)
    group_pairwise_corr(bottom_antennas_matrix, fname_bottom,number_of_groups=10)
#    group_pairwise_corr(antennas_matrix, fname_all,number_of_groups=10)

def std_margines_lsoa_dep_index(antennas_matrix,antennas_std, antennas_indices, antennas_lsoa, lsoa_dpr, dpr_name, fname):
    top,middle,bottom = group_antennas(antennas_std)
    top_aids = [antennas_indices[k] for k in  top.iterkeys()]
    bottom_aids = [antennas_indices[k] for k in bottom.iterkeys()]
    top_dpr = [lsoa_dpr[antennas_lsoa[i].lower()] for i in top_aids]
    bottom_dpr = [lsoa_dpr[antennas_lsoa[i].lower()] for i in bottom_aids]
    allv = lsoa_dpr.values()
    dpr_name = dpr_name.split(" (")[0]
    r = {'top': top_dpr, "bottom": bottom_dpr,'all':allv}
    for pname, data in r.iteritems():
        fig, ax = plt.subplots()
        ax.hist(data,10,alpha=0.7)
        ax.set_title("{0}: {1} 10%".format(dpr_name,pname))
        ax.set_xlabel("visited LSOAs deprivation level")

        plt.savefig(os.path.join(output_path_plots,"dpr_hist","{0}_{1}_{2}.png".format(fname,pname,dpr_name)), bbox_inches="tight")
        plt.close() 


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



def group_antennas(antennas_gini,ratio=0.1):
    """ 
    groups antennas to top and bottom 10% and the rest
    """
    valid_aids = antennas_gini.keys()
    values = sorted(set(antennas_gini.values()),reverse=True)
    l = int(len(values)*ratio) #10%
    
    top_antennas = [(k,v) for k,v in antennas_gini.iteritems() if v>=values[l]]    
    bottom_antennas = [(k,v) for k,v in antennas_gini.iteritems() if v<=values[-l]]    
    middle_antennas = [(None,None)]
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
