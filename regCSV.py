import sys
import csv
from operator import itemgetter
import os
import subprocess

from collections import defaultdict,Counter

import numpy as np

class RegCSV(object):
    def __init__(self, theta=0.05, decimals=3, exp=True):
        '''
        inpute:
            theta: significance level. p-values less than <theta> are not significance
            decimals: round float values up to <decimals> decimal values
        '''
        self.model_objs = {}
        self.model_fits = {}
        self.stats_test = None
        self.theta = theta
        self.decimals = decimals
        self.exp = exp

    def _get_stats(self, model_fit):
        '''
        check if the model computed z-stat or t-stat
        '''
        if hasattr(model_fit,'zvalues'):
            stats = model_fit.zvalues
            self.stats_test = 'z'
        else:
            stats = model_fit.tvalues
            self.stats_test = 't'
        return np.round(stats, self.decimals)


    def _reformat(self, model_objs, model_fits=None):
        '''
        summaries the fit statistics by keeping only coefficient names, coefficient values, z/t-statistics and the 
        indication of significance of the coefficient using *
        input:
            model_objs: dictionary. keys correspond to model name and values are statsmodel regression model object
            model_fits: dictionary. keys correspond to model name and values to the fited regression model object
        output:
            resutlt: nested dictionary where for first key(=model name), there is dictionary in which keys correponds to the model coefficient names
            and values string of <coefficient values(z/t statistics)*>.
        Note: 
            <*> appeares only for coefficient with statistical significance

        '''
        #if model fites did not pre-computed, compute them now
        if not model_fits:
            for model_name, model in model_objs.iteritems():
                self.model_fits[model_name] =  model.fit()
        else:
            self.model_fits = model_fits
        
        result = defaultdict(dict)
        for model_name,model in model_objs.iteritems():
            model_summary = {}
            model_fit = model_fits[model_name]
            coeff_names = model.exog_names
            params = model_fit.params.copy()
            if self.exp:
                params = np.exp(params)
            params = np.round(params, self.decimals)
            pvalues = model_fit.pvalues
            stats = self._get_stats(model_fit)
            
            for i in xrange(len(stats)):
                #order is important
                if pvalues[i]<=0.001:
                    temp = '{0}({1})***'.format(params[i], stats[i])
                elif pvalues[i]<=0.01:
                    temp = '{0}({1})**'.format(params[i], stats[i])
                elif pvalues[i]<=self.theta:
                    temp = '{0}({1})*'.format(params[i], stats[i])
                else:
                    temp = '{0}({1})'.format(params[i], stats[i]) #not significANT
                model_summary[coeff_names[i]] = temp

            result[model_name] = model_summary
        return result
    
    def _short_interaction_term(self,coeff):
#        print coeff
        try:
            temp,temp2 = coeff.split(',')
            T = temp2[temp2.find('[')+3]
            if ':' in temp:
                var1,temp = temp.split(':')
                var2 = temp.strip('C(')
                new_var = "{0}*{1}[{2}]".format(var1,var2,T)
            else:#for gender
                var2 = temp.strip('C(')
                new_var = "{0}[{1}]".format(var2,T)
        except Exception as err:
            new_var = coeff
#        print coeff
#        print new_var
        return new_var

    def _get_coeffs(self, result):
        '''
        return the coefficient from the largest set
        '''
#        print 'here'
        coeffs = []
        coeff_repeated = []
        for model_name, stats in result.iteritems():
            temp = sorted(stats.keys())
            coeff_repeated.extend(temp)
            #coeffs.update(temp)
        coeff_count = Counter(coeff_repeated)
        coeff_count_s = sorted(coeff_count.iteritems(), key=itemgetter(0))
        for c in coeff_count_s:
            coeff_name1 = c[0]
#            print '-------------'
#            print coeff_name1
            if 'C(gender' in coeff_name1:
#                print 'interaction'
                coeff_name2 = self._short_interaction_term(coeff_name1)
                coeff_name2 = coeff_name2.replace('_',' ')
                coeffs.append((coeff_name1,coeff_name2))
#            elif 'C(gender':
#                coeff_name2 = 'gender[f]'
#                coeffs.append((coeff_name1,coeff_name2))
            elif 'career_age_t' in coeff_name1:
                pass
            elif 'year' in coeff_name1:
                pass
            elif 'decade' in coeff_name1:
                coeffs.append((coeff_name1,coeff_name1))
#                pass
            else:
                coeff_name2 = coeff_name1.replace('_',' ')
                coeffs.append((coeff_name1,coeff_name2))
        #print coeffs
        return coeffs

    def _vars_as_col(self, fpath, result, wmod, delimiter):
        '''
        put variables names in columns and model names in rows
        '''
        print 'variable as columns'
        def make_row(model_name):
            row = [model_name]
            for coeff in coeff_names:
                try:
                    val = result[model_name][coeff[0]]
                except Exception as err:
                    val = None
                finally:
                    row.append(val)
            return row
            
        with open(fpath,wmod) as f:
            csv_writer = csv.writer(f,delimiter=delimiter)
            #coeff_names = sorted(result[model_names[0]].keys())
            coeff_names = self._get_coeffs(result)
            print coeff_names
            header = [c[1] for c in coeff_names]
            header.insert(0,'')
#            print header
            csv_writer.writerow(header)
            model_names = sorted(result.keys())
            for model_name in model_names: 
                row = make_row(model_name)
                csv_writer.writerow(row)
            csv_writer.writerow([''])

    def _vars_as_row(self, fpath, result, wmod, delimiter):
        '''
        put variables names in rows and model names in columns
        '''
        print 'variable as rows'
        def make_row(coeff_name):
            row = [coeff_name[1]]
            for model_name in model_names:
                try:
                    val = result[model_name][coeff_name[0]]
                except Exception as err:
                    val = None
                finally:
                    row.append(val)
            return row
            
        with open(fpath,wmod) as f:
            csv_writer = csv.writer(f,delimiter=delimiter)
            model_names = sorted(result.keys())
            header = list(model_names)
            header.insert(0,'')
            csv_writer.writerow(header)
            #coeff_names = sorted(result[model_names[0]].keys())
            coeff_names = self._get_coeffs(result)
#            print coeff_names
            for coeff in coeff_names: 
                row = make_row(coeff)
                csv_writer.writerow(row)
            csv_writer.writerow([''])
    
    def addStats(self,model_stats,stats_name ,var_pos, fpath,delimiter):
        '''
        model_stats: a list containing a dictionary with name and value of specific stats
        '''
#        print fpath
        if var_pos == 'row':
            with open(fpath,'ab') as f:
                csv_writer = csv.writer(f,delimiter=delimiter)
                #for stats in model_stats:
                for name in stats_name:
                    values = [model[name] for model in model_stats]
                    values.insert(0,name)
                    csv_writer.writerow(values)
        else:
            sys.exit('not implemented')

    def save_as_csv(self, fpath, model_objs, model_fit_objs=None, vars_pos=None,wmod='wb', delimiter='#'):
        ''' 
        main function to create a csv file of regression summary
        input:
            fpath: path to save the csv file
            model_objs: dictionary with keys as model name and values as statsmodel regression model object
            model_fit_objs: user can pass the regression fit object to the function otherwise (None value)they will computed later.
            var_pos: can take 3 different values: "row","col" and None. if "row", variable names are written as the row,if "col" they will 
            be written as column. if None each model will be written seperately in the csv file
            wmod: "wb" or "ab". if "wb" a new file be created. if "ab" current output will be appended to a pre-existing file. 

        '''
        result = self._reformat(model_objs, model_fit_objs)
        if vars_pos == 'row':
            self._vars_as_row(fpath,result,wmod, delimiter)
        elif vars_pos == 'col':
            self._vars_as_col(fpath, result,wmod,delimiter)
        else:
            with open(fpath,wmod) as f:
                csv_writer = csv.writer(f,delimiter=delimiter)
                for model_name, stats in result.iteritems(): 
                    csv_writer.writerow([''])
                    csv_writer.writerow(['\n***',model_name,'***\n'])
                    csv_writer.writerow(['','coeff({0})'.format(self.stats_test)])
                    coeff_names_sorted = sorted(stats.keys())
                    for coeff_name in coeff_names_sorted:
                        csv_writer.writerow([coeff_name,stats[coeff_name]])
        
#        self.csv_to_pdf(fpath)
    
    def csv_to_pdf(self,csv):
        fpath = csv.split('.')[0]
        tex = "{0}.tex".format(fpath)
#        print tex
#        command = "csv2latex -s s --colorrows 0.75 --reduce 2 {0} > {1}".format(csv,tex)
#        os.system(command)
        a= subprocess.check_output(["csv2latex","-s","s","--colorrows", "0.75","--reduce","2"
            ,csv,">",tex,"&&","pdflatex",tex])
#        r = a.communicate()[0]
        with open(tex,'wb') as f:
            a= subprocess.call(["csv2latex","-s","s","--colorrows", "0.75","--reduce","2"
                ,csv,">",tex],stdout=f)
        
        with open(tex,'wb') as f:
            a= subprocess.call(["csv2latex","-s","s","--colorrows", "0.75","--reduce","2"
                ,csv,">",tex],stdout=f)
#
#        command = "pdflatex {0}".format(tex)
#        os.system(command)
        

    def add_formulas(self,fpath,formulas, delimiter):
        with open(fpath,'ab') as f:
            csv_writer = csv.writer(f,delimiter=delimiter)
            for i, formula in enumerate(formulas): 
                csv_writer.writerow([i,formula])


def get_example_linear(seed):
    import statsmodels.api as sm

    np.random.seed(seed)
    nsample = 100
    x = np.linspace(0, 10, 100)
    X = np.column_stack((x, x**2))
    beta = np.array([1, 0.1, 10])
    e = np.random.normal(size=nsample)
    X = sm.add_constant(X)
    y = np.dot(X, beta) + e
    model = sm.OLS(y, X)
    fit = model.fit()
    return model,fit
def get_example_nonlinear():
    import statsmodels.api as sm
    nsample = 50
    sig = 0.5
    x = np.linspace(0, 20, nsample)
    X = np.column_stack((x, np.sin(x), (x-5)**2, np.ones(nsample)))
    beta = [0.5, 0.5, -0.02, 5.]

    y_true = np.dot(X, beta)
    y = y_true + sig * np.random.normal(size=nsample)
    res = sm.OLS(y, X)
    return res, res.fit()
if __name__ == '__main__':
    model1,fit1 = get_example_linear(1)
#    print fit1.summary()
    model2,fit2 = get_example_nonlinear()
#    print fit2.summary()
    models = {'linear1':model1,'linear2':model2}
    model_fits = {'linear1':fit1,'linear2':fit2}
    reg_csv = RegCSV()
    fpath = "regcsv_test.csv"
    reg_csv.save_as_csv(fpath,models,model_fits, vars_pos=None,wmod='ab')