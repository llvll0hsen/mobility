import os
import sys
from collections import defaultdict,Counter
import dill
import traceback 

import statsmodels.api as sm
import pandas as pd
from statsmodels.formula.api import logit
from patsy.contrasts import Treatment
import numpy as np

import util
from antenna_analysis import group_antennas,comp_prob_std
from regCSV import RegCSV

from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)
def prepare_data(antennas_matrix,antennas_indices,antenna_lsoa_mapping, valid_aids,fname):
    antennas_std = {}
    i = 0
    for aidx in valid_aids:
        aid = antennas_indices[aidx]
        vec = antennas_matrix[aidx,:]
        try:
            temp_std = comp_prob_std(vec)
            antennas_std[aid] = temp_std
        except Exception as err:
            traceback.print_exc()
            pass
    top,middle,bottom = group_antennas(antennas_std)
    aids_record = {k:[1] for k in top.iterkeys()}
    for k in bottom.iterkeys():
        aids_record[k] = [0]
    
    
    
    factors = ["LSOA code (2011)","IMD Decile (where 1 is most deprived 10% of LSOAs)", "Income Decile (where 1 is most deprived 10% of LSOAs)","Employment Decile (where 1 is most deprived 10% of LSOAs)","Education, Skills and Training Decile (where 1 is most deprived 10% of LSOAs)","Health Deprivation and Disability Decile (where 1 is most deprived 10% of LSOAs)",
            "Crime Decile (where 1 is most deprived 10% of LSOAs)","Barriers to Housing and Services Decile (where 1 is most deprived 10% of LSOAs)",
            "Living Environment Decile (where 1 is most deprived 10% of LSOAs)"]

    valid_lsoas = antenna_lsoa_mapping.values()
    
    df = pd.read_excel(os.path.join(util.census_data_fpath,"london","deprivation_london.xls"),sheet_name="IMD 2015")
    df = df[factors].dropna()
    df = df.set_index("LSOA code (2011)")
    df = df.loc[df.index.isin(valid_lsoas)]

    
    dpr_dict = df.to_dict()
    col_names = [n.split(' (')[0].replace(" Decile", "").replace(",","").replace(" ","_") for n in factors[1:]]
    for aid in aids_record.iterkeys():
        try:
            lsoa =  antenna_lsoa_mapping[aid]
            for fname in factors[1:]:
                val = dpr_dict[fname][lsoa]
                aids_record[aid].append(val)
        except Exception as err:
            pass
            #traceback.print_exc()
        
    df_out = pd.DataFrame.from_dict(aids_record,'index')
    col_names.insert(0,"top")
    df_out.columns = col_names
#    print col_names
#    print df_out.head(1)
#    dill.dump(df_out,open(os.path.join(util.output_path_files,"mobility","reg_data","{0}.dill".format(fname)),"wb"))
    return df_out

def apply_reg(antennas_matrix,antennas_indices,antenna_lsoa_mapping, valid_aids,fname):
    data = prepare_data(antennas_matrix,antennas_indices,antenna_lsoa_mapping, valid_aids,fname)
     
#    model = logit('top ~ C(Income,Treatment(reference=1)) + C(Employment,Treatment(reference=1)) + C(Education_Skills_and_Training,Treatment(reference=1))+C(Health_Deprivation_and_Disability,Treatment(reference=1)) + C(Crime,Treatment(reference=1)) + C(Barriers_to_Housing_and_Services,Treatment(reference=1)) + C(Living_Environment,Treatment(reference=1))',data).fit()#(maxiter=100)
    model = logit('top ~ Income + Employment + Education_Skills_and_Training+Health_Deprivation_and_Disability+Crime+Barriers_to_Housing_and_Services+Living_Environment' ,data).fit()
    print '\n----------------{0}------------------'.format(fname)
    #model.params   =  model.params.apply(np.exp)
    conf  = np.exp(model.conf_int())
    conf['OR'] =np.exp( model.params)
    conf['pvalue'] = model.pvalues

    conf.columns = ['2.5%', '97.5%', 'OR','P']

    print conf
