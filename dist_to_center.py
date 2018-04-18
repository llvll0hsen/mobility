import os
import sys
from collections import defaultdict

import dill
from scipy import stats
from vincenty import vincenty
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pylab as plt

from util import output_path_files, census_data_fpath ,output_path_plots,connect_mongodb, reverse_geo_mongodb,get_depr_factor


lat_c,lon_c = (51.509865, -0.118092)
aid_lsoa = dill.load(open(os.path.join(output_path_files,"mobility","antenna_lsoa_london_only_new.dill"),"rb"))
aid_dist = dill.load(open(os.path.join(output_path_files,"mobility","antenna_dis2cent.dill"),"rb"))
def dist_corr(aid_std,fname):
#    dpr = get_depr_factor("IMD Decile (where 1 is most deprived 10% of LSOAs)")

    
#    aid_dpr = {aid:dpr[lsoa.lower()] for aid,lsoa in aid_lsoa.iteritems()}
    temp = [(aid_dist[aid],std) for aid,std in aid_std.iteritems()]
    a,b = zip(*temp)
    r = stats.pearsonr(a,b)
    
    print "{0}: {1}".format(fname,r) #(0.18562076663754282, 3.3548610751062493e-159)
    #old (0.177192130351583, 3.8633695795760655e-195)
def comp_distance():
    result = {} 
    antenna_loc_london = dill.load(open(os.path.join(output_path_files,"mobility","antenna_loc_london_only_new.dill"),"rb"))
#    del antenna_loc_london["LACOD-SAC"]
    for aid, loc in antenna_loc_london.iteritems():
        lon,lat = loc
        print aid,loc
        d = vincenty((lon_c,lat_c),(float(lon),float(lat)))
        result[aid] = d
#        print loc,d
    dill.dump(result, open(os.path.join(output_path_files,"mobility","antenna_dis2cent.dill"),"wb"))

def lsoas():
    from shapely import wkt
    collection,client = connect_mongodb()
    for doc in collection.find({},limit=2):

        print doc['properties']['LAD11CD']
        print   doc['geometry']#['coordinates']

    client.close()

def plot():
    files = os.listdir(os.path.join(output_path_files,"mobility","dist_center_corr"))
    time_labels = ["00-04","04-08","08-12","12-16","16-20","20-24"]#[2:]
    weekday_labels = ["Mon","Tue", "Wed", "Thu","Fri", "Sat","Sun"]
    d = defaultdict(dict)
    for fname in files:
        f = open(os.path.join(output_path_files,"mobility","dist_center_corr",fname),"rb")
        for line in f:
            fname, corr_info = line.strip().split(": ")
            ts = fname.split('_')[-1]
            val, pvalue = corr_info.strip("()").split(",")
            d[ts] = (float(val),float(pvalue))

    times = {"daily":time_labels,"weekly":weekday_labels}
    for scale,timeslots in times.iteritems():
        x = range(len(timeslots))
        fig, ax = plt.subplots()
        temp_vals = [d[t][0] for t in timeslots]
        ax.plot(x, temp_vals)
        ax.set_xticks(x)
        ax.set_xticklabels(timeslots,rotation=45)
        ax.scatter(x,temp_vals, s=50,marker="o")
        for i,t in enumerate(timeslots):
            pvalue = d[t][1]
            print pvalue
            if pvalue>=0.01:
                ax.scatter(i,temp_vals[i], s=40,marker="s",c="black")

        ax.set_ylabel("std-dist corr") 
#        ax.legend(loc='upper left',ncol=1,frameon=True,bbox_to_anchor=(1.04, 1))#fancybox=True)
        plt.savefig(os.path.join(output_path_plots,"mobility","dist_center_corr","{0}.png".format(scale)), bbox_inches="tight")
        plt.close() 
if __name__ == '__main__':
#    comp_distance()
#    dist_corr()
#    plot()
    lsoas()
