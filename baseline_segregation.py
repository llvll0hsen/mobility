import os
import sys
import json
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import dill

"""
baseline is computed by looking at the segregation level of each district by comparing it to it's appriximity.

in this setting t_m would be the total number of people in the section m and it's neighbors

"""


def get_sector_population(aid):
    import random
    return random.choice(range(10,500))
if __name__ == '__main__':
    antenna_count = pd.read_pickle("data/count_by_antenna.pkl")
    ordinal_seg = dill.load(open('sectorDiversity_ordinal.dill','rb'))
    antennas_ne = dill.load(open(os.path.join('output','files','antennas_ne_polygon.dill'),'rb'))
    antenna_imd = pd.read_pickle("Antenna_IMD.pkl")
    antenna_imd['IMD_DEC'] = antenna_imd['decile'].round().astype(int)
    k = sorted(antenna_imd.IMD_DEC.unique().tolist())
    K = len(k)-1
    
    
    mk_raw = defaultdict(lambda:defaultdict(float))
    Cmk = defaultdict(list)
    m_var = {}    
    miss = 0
    for aid, ne_aids in antennas_ne.items():
        ne_aids.append(aid) #we should also count number of people in section m as well
        td = 0
        for i in ne_aids:
            try:
                imd = int(antenna_imd[antenna_imd.SectorID==i]['IMD_DEC'])
                ucount = antenna_count.loc[i]['user_id']
                mk_raw[aid][imd]+=ucount
                td+=ucount
            except Exception as err:
#                print(antenna_imd[antenna_imd.SectorID==i]['IMD_DEC'])
#                print(err)
                pass
        try:
            mk = np.round(np.cumsum([mk_raw[aid][j]/td for j in k]),1)
            m_var[aid] = (1/(K)) * sum([2*np.sqrt(mk[c]*(1-mk[c])) for c in range(K)])
        except  Exception as err:
#            print(err)
            miss+=1
            continue

    print(miss,len(antennas_ne))
            #*get_sector_population(i)  
#        mk_raw[aid] = Counter([int(antenna_imd[antenna_imd.SectorID==i]['IMD_DEC'])*get_sector_population(i) for i in ne_aids])
#        td = sum(mk_raw[aid].values())
#        mk = np.round(np.cumsum([mk_raw[aid][i]/td for i in k]),1)
#        m_var[aid] = (1/(K)) * sum([2*np.sqrt(mk[c]*(1-mk[c])) for c in range(K)])
#
    with  open('sectorDiversity_ordinal_baseline.dill','wb') as handle:
        dill.dump(m_var,handle)
#    
        
    
        


