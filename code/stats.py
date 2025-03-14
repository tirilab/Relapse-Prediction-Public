import numpy as np
from scipy.stats import kstest
import statsmodels.stats.multitest as multitest
from sklearn.feature_selection import chi2

def statistical_test(df, continuous_var, categorical_var):
       
       # generate binary outcome: OF vs. nonOF
       # df = pd.read_csv(input_file, index_col=0)
       # df['outcome'] = df[['acute kidney failure', 'acute respiratory failure', 'acute heart failure']].sum(axis=1)
       df['outcome'] = [0 if i ==0 else 1 for i in df.outcome]
       df0 = df[df.outcome == 0]
       df1 = df[df.outcome == 1]

       cont_stats = []
       test_result_cont = []
       p_value = []
       if len(continuous_var) > 0:
              for var in continuous_var:
                     # summary stats of continuous variable
                     cont_stats.append([round(df0[var].mean(),2), round(df0[var].std(),2), 
                                        round(df1[var].mean(),2), round(df1[var].std(),2)])
                     test_result_cont.append(round(kstest(df0[var], df1[var]).statistic, 2))   
                     p_value.append(round(kstest(df0[var], df1[var]).pvalue, 3))

       cat_stats = []
       test_result_cat = []
       if len(categorical_var) > 0:
              # summary stats of categorical variable
              for var in categorical_var:
                     cat_stats.append([df0[var].sum(), df0[var].mean()*100, df1[var].sum(), df1[var].mean()*100])
              
                     # categorical: chi-square test 
                     _, test_result_cat = chi2(df[categorical_var], df['outcome'])

       # adjust p-values through bonferroni
       _, pval_adj_cont, _, _ = multitest.multipletests(p_value, method='fdr_bh') 
       # pval_adj_cont = np.multiply(test_result_cont, len(continuous_var))
       pval_adj_cat = np.multiply(test_result_cat, len(categorical_var))

       return test_result_cont, pval_adj_cont, cont_stats
