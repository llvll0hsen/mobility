import numpy as np
import os
import pandas as pd
from collections import Counter, defaultdict
import dill
import warnings
import sys

def get_Cj(df):
    '''
    ordinal var
    c_k indicate the cumulative propotion of the population who are in category k or below
    '''
    hist = Counter(df.IMD_DEC_Source.tolist())
    temp  = [float(hist[i]) for i in range(1,11)]
    C = np.cumsum(temp)
    t = C[-1]
    C = [i/t for i in C]

    return C
  
def get_Cmk(df):
 
    ''' 
    within each unordered category m, denote the cumulative proportion of the population in m who are in ordered 
    category k or below as Cmk
    '''
    dest = df.dest.unique().tolist()
    k = sorted(df.IMD_DEC_Source.unique().tolist())
    dest_group = df.groupby('dest')['freq','IMD_DEC_Source']
    
    mk_raw = defaultdict(lambda:defaultdict(float))
    Cmk = defaultdict(list)
#    Cmk2 = defaultdict(list)
    
    for d in dest: 
        td=0.
        temp = dest_group.get_group(d).set_index('IMD_DEC_Source')
        for row in temp.itertuples():
            print(row)
            mk_raw[d][row[0]]+=row[1]
            td+=row[1]
        Cmk[d] = np.round(np.cumsum([mk_raw[d][i]/td for i in k]),1)
#        Cmk2[d] = np.cumsum([mk_raw[d][i] for i in k])
        
    return Cmk

def compute_variation(Cmk):
    '''
    use ordinal variation ratio
    '''
    K = 9
    m_var = defaultdict(float)
    warnings.simplefilter('error')
    for m,mk in Cmk.items():
        m_var[m] = (1/(K)) * sum([2*np.sqrt(mk[c]*(1-mk[c])) for c in range(K)])
#        m_var[m] =  (1/(K)) * sum([4*mk[c]*(1-mk[c]) for c in range(K)])
    return m_var

def diversity_by_deprivation():
    '''
    compute diversity measure for different group of deprivation
    1-4
    5-7
    8-10
    '''
    df = pd.read_pickle("Antenna_IMD.pkl")
    group_low = df[(df.decile>=1) &  (df.decile<4)]['SectorID']
    group_middle = df[(df.decile>=5) &  (df.decile<7)]['SectorID']
    group_high = df[(df.decile>=8) &  (df.decile<10)]['SectorID']



if __name__ == '__main__':
    
    df = pd.read_pickle('antenna_src_dec_filtered.pkl')
    antennas = df.dest.unique()
    levels = df.IMD_DEC_Dest.unique()
    K = len(levels) #number of deprivation levels - ordinal variable
    M = len(antennas) #numbe or neighborhoods - non-ordinal variable
    df = pd.read_pickle('antenna_src_dec_filtered.pkl')
    Cmk = get_Cmk(df)
    sys.exit()
    r = compute_variation(Cmk)
#    
    with  open('sectorDiversity_ordinal.dill','wb') as handle:
    
        dill.dump(r,handle)
