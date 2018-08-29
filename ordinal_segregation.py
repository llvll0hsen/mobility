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
#    print(k)
#    k = range(1,10)
    dest_group = df.groupby('dest')['freq','IMD_DEC_Source']
    
    mk_raw = defaultdict(lambda:defaultdict(float))
    Cmk = defaultdict(list)
    Cmk2 = defaultdict(list)
    
    for d in dest: 
        td=0.
        temp = dest_group.get_group(d).set_index('IMD_DEC_Source')
        for row in temp.itertuples():
#            print(row)
            mk_raw[d][row[0]]+=row[1]
            td+=row[1]
        Cmk[d] = np.round(np.cumsum([mk_raw[d][i]/td for i in k]),1)
        Cmk2[d] = np.cumsum([mk_raw[d][i] for i in k])
        
        a = sorted(Cmk[d])
        if a[-1]>1:
            print(a)
#            for i in k:
#                if mk_raw[d][i]>td:
#                    print(i,mk_raw[d][i],td)
#            print(a)
#            break
            
#        tm = Cmk[d][-1]
#        if tm>1:
#            print(d,tm,Cmk2[d][-1]/td)

#        Cmk[d] = [i/tm for i in Cmk[d]]
#    print(Cmk['21008-35800'])
    return Cmk

def ordinal_variation_ratio(c):
    pass

 
def compute_variation(Cmk):
    '''
    use ordinal variation ratio
    '''
    K = 9
    m_var = defaultdict(float)
    warnings.simplefilter('error')
    mk = Cmk['22020-31140']
    print(mk)
    print(([4*mk[c]*(1-mk[c]) for c in range(K)]))
    print(1/(K)*sum([4*mk[c]*(1-mk[c]) for c in range(K)]))
#    print(([4*c*(1-c) for c in range(K)]))
#    print(Cmk['21007-20032'])
#    print(Cmk['21007-30584'])

#    print([(mk[c],1-mk[c],np.sqrt(mk[c]*(1-mk[c]))) for c in range(K)])
#    m = (1/(K-1)) * sum([2*np.sqrt(mk[c]*(1-mk[c])) for c in range(K)])
#    print(m)
#    sys.exit()
    for m,mk in Cmk.items():
#        print(mk)
#        print([(mk[c],1-mk[c],mk[c]*(1-mk[c])) for c in range(K)])
#        print(m)
#        print('---')
        m_var[m] = (1/(K)) * sum([2*np.sqrt(mk[c]*(1-mk[c])) for c in range(K)])
#        m_var[m] =  (1/(K)) * sum([4*mk[c]*(1-mk[c]) for c in range(K)])
#        if m_var[m]>1:
#            print(m,m_var[m])
#            break
    return m_var

if __name__ == '__main__':
    
    df = pd.read_pickle('antenna_src_dec_filtered.pkl')
    antennas = df.dest.unique()
    levels = df.IMD_DEC_Dest.unique()
    K = len(levels) #number of deprivation levels - ordinal variable
    M = len(antennas) #numbe or neighborhoods - non-ordinal variable
    df = pd.read_pickle('antenna_src_dec_filtered.pkl')
    Cmk = get_Cmk(df)
    r = compute_variation(Cmk)
#    
    with  open('sectorDiversity_ordinal.dill','wb') as handle:
    
        dill.dump(r,handle)
