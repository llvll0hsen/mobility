import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pylab as plt
import dill

from util import output_path_files, output_path_plots,connect_monogdb, reverse_geo_mongodb,mobility_path,census_data_fpath,get_depr_factor

output_path_plots = os.path.join(output_path_plots,"mobility")
output_path_files = os.path.join(output_path_files,"mobility")



if __name__ == "__main__":
    files = os.listdir(os.path.join(output_path_files,"dpr_corr"))
    factors = ["IMD","Children and Young People Sub-domain Decile (where 1 is most deprived 10% of LSOAs)","Adult Skills Sub-domain Decile (where 1 is most deprived 10% of LSOAs)","Geographical Barriers Sub-domain Decile (where 1 is most deprived 10% of LSOAs)","Wider Barriers Sub-domain Decile (where 1 is most deprived 10% of LSOAs)","Indoors Sub-domain Decile (where 1 is most deprived 10% of LSOAs)","Outdoors Sub-domain Decile (where 1 is most deprived 10% of LSOAs)"]
    d = defaultdict(dict)

    for fname in files:
        t = fname.split("_")[2]
        f = open(os.path.join(output_path_files,"dpr_corr",fname),"rb")
        for line in f:
            factor, corr_info = line.strip().split("\t")
            val, pvalue = corr_info.strip("()").split(",")
            d[factor][t] = (float(val),float(pvalue))
    print d
    time_labels = ["00-04","04-08","08-12","12-16","16-20","20-24"]#[2:]
    weekday_labels = ["Mon","Tue", "Wed", "Thu","Fri", "Sat","Sun"]
#    dill.dump(d,open(os.path.join(output_path_files,"dpr_corr.dill"),"wb"))
    times = {"daily":time_labels,"weekly":weekday_labels}

    for scale,timeslots in times.iteritems():
        x = range(len(timeslots))
        fig, ax = plt.subplots()
        for factor, t_vals in d.iteritems():
            factor = factor.split(" (")[0]
            temp_vals = [t_vals[t][0] for t in timeslots]
            ax.plot(x, temp_vals,label=factor)
            ax.set_xticks(x)
            ax.set_xticklabels(timeslots,rotation=45)
            ax.scatter(x,temp_vals, s=50,marker="o")
            for i,t in enumerate(timeslots):
                pvalue = t_vals[t][1]
                if pvalue<=0.01:
                    ax.scatter(i,temp_vals[i], s=40,marker="s",c="black")

        ax.set_ylabel("std-depr corr") 
        ax.legend(loc='upper left',ncol=1,frameon=True,bbox_to_anchor=(1.04, 1))#fancybox=True)
        plt.savefig(os.path.join(output_path_plots,"dpr_corr","{0}.png".format(scale)), bbox_inches="tight")
        plt.close() 


