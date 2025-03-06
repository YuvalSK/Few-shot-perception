# -*- coding: utf-8 -*-
"""
Few-shot learning of visual length
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter, FixedLocator
import seaborn as sns

import statsmodels.api as sm
from statsmodels.stats.anova import AnovaRM
from scipy.stats import linregress, sem, ks_2samp, wilcoxon, bartlett, fligner, levene, f_oneway, norm, ttest_ind, ttest_1samp, ttest_ind_from_stats, permutation_test, spearmanr, mannwhitneyu, shapiro
from pingouin import bayesfactor_ttest, welch_anova, ttest

def med_statistic(x, y):
    return np.median(x) - np.median(y)

def avg_statistic(x, y):
    return np.mean(x) - np.mean(y)

def spearmanr_statistic(x, y):
    return spearmanr(x, y)[0]

sns.set_style("whitegrid")

df = pd.read_csv('Data/Experiment 9 - preregistered/merged - filtered.csv')

#parsing the data
def parse_data(df):
    participant = []
    trial = []
    condition = []
    response = []
    rts = []
    
    for c, i in enumerate(df["participant #"]):
        if np.isfinite(df["4.Line Length.0.answer"][c]) & np.isfinite(df["2.Line Length.0.answer"][c]) & np.isfinite(df["0.Line Length.0.answer"][c]):
            participant.append(i)
            participant.append(i)
            participant.append(i)
            trial.append("T1")
            trial.append("T2")
            trial.append("T3")
            condition.append(df["Condition"][c])
            condition.append(df["Condition"][c])
            condition.append(df["Condition"][c])
            response.append(df["0.Line Length.0.answer"][c])
            response.append(df["2.Line Length.0.answer"][c])
            response.append(df["4.Line Length.0.answer"][c])
            rts.append(df["0.Line Length.0.rt"][c])
            rts.append(df["2.Line Length.0.rt"][c])
            rts.append(df["4.Line Length.0.rt"][c])
    
    
    df_response = pd.DataFrame({"id": participant, "trial": trial, "condition": condition, "response": response, "RT": rts})
    return df_response

df_response = parse_data(df)
conditions_names = {0: 'Pooled',
                    1:'F-15',
                    2: 'F-30',
                    3: 'F-60',
                    4: 'A-up [20-40-20]',
                    5: 'A-down [40-20-40]',
                    6: 'F-20',
                    7: 'F-40',
                    8: 'A-down [15-10-15]'}

'''
analysis by prior groups
'''

#by log scale
def main(file):
    df = pd.read_csv('Data/Experiment 9 - preregistered/merged - filtered.csv')
    
    options = [2, 7, 3]
    df_stp = df[df['Condition'].isin(options)]
    #change ms units to s 
    df_stp['0.Line Length.0.rt'] = df_stp['0.Line Length.0.rt'] / 1000
    df_stp['2.Line Length.0.rt'] = df_stp['2.Line Length.0.rt'] / 1000
    df_stp['4.Line Length.0.rt'] = df_stp['4.Line Length.0.rt'] / 1000

    fig = plt.figure()
    gs = fig.add_gridspec(2,3)
    ax11 = fig.add_subplot(gs[0,0])
    ax12 = fig.add_subplot(gs[0,1])
    ax13 = fig.add_subplot(gs[0,2])
    
    ax21 = fig.add_subplot(gs[1,0])
    ax22 = fig.add_subplot(gs[1,1])
    ax23 = fig.add_subplot(gs[1,2])
    
    line_w = 2.5
    
    palette = {30:'lightgray',  
               40:'gray',
               60:'k'}
    
    df_i = df_stp.loc[df_stp['STP']=='Internal']
    df_long = df_i.melt(id_vars=['participant #', 'scale'], 
                       value_vars=['T1 ratio', 'T2 ratio', 'T3 ratio'], 
                       var_name='Measure', 
                       value_name='Value')
    
    #df_long['Value'] = np.log10(df_long['Value'])
    sns.pointplot(data=df_long, x='Measure', y='Value', 
                  hue='scale', 
                  hue_order=[30, 40, 60],
                  estimator=np.mean,
                  errorbar='se',
                  linewidth=line_w,
                  err_kws={"color": "k", 'linewidth': 0.5},
                  capsize = 0.1,
                  markers='o',
                  palette=palette,
                  dodge=True, 
                  ax=ax11)
    
    ax11.set_xlabel("")
    ax11.set_xticks([0, 1, 2], ['','',''])
    
    ax11.set_yscale('log')
    ax11.set_ylim(0.15, 1.1)
    ax11.set_yticks([0.5, 1])  # Explicitly set desired ticks
    ax11.yaxis.set_major_formatter(ScalarFormatter())  # Force numbers
    ax11.yaxis.set_minor_locator(plt.NullLocator())  # Remove extra minor ticks
    
    ax11.axhline(y=1, c='k', linestyle='--')
    ax11.set_ylabel("")
    ax11.legend(title='Scale', loc='lower right', 
               title_fontsize='small', 
               fontsize='x-small')
    
    label1 = 'Estimate to stimulus\n ratio [A.U.]'
    ax11.set_ylabel(label1)
    sns.despine(ax=ax11, top = True, right = True)
    
    df_long = df_i.melt(id_vars=['participant #', 'scale'], 
                        value_vars=['0.Line Length.0.rt', '2.Line Length.0.rt', '4.Line Length.0.rt'], 
                       var_name='Measure', 
                       value_name='Value')
    
    sns.pointplot(data=df_long, x='Measure', y='Value', 
                  hue='scale', 
                  hue_order=[30, 40, 60],
                  estimator=np.mean,
                  errorbar='se',
                  linewidth=line_w,
                  err_kws={"color": "k", 'linewidth': 0.5},
                  capsize = 0.1,
                  palette=palette,
                  dodge=True, markers='o',
                  ax=ax21)
    
    ax21.get_legend().remove()
    ax21.set_xlabel("Internal")
    ax21.set_xticks([0, 1, 2], ['T1','T2','T3'])    
    ax21.set_ylim(1, 6)
    ax21.set_yticks(np.arange(1, 7, 1))
    sns.despine(ax=ax21, top = True, right = True)
    label1 = 'Response time\n[s]'
    ax21.set_ylabel(label1)
    
    df_e = df_stp.loc[df_stp['STP']=='External']
    df_long = df_e.melt(id_vars=['participant #', 'scale'], 
                       value_vars=['T1 ratio', 'T2 ratio', 'T3 ratio'], 
                       var_name='Measure', 
                       value_name='Value')
    
    sns.pointplot(data=df_long, x='Measure', y='Value', 
                  hue='scale', 
                  hue_order=[30,40, 60],
                  estimator=np.mean,
                  errorbar='se',
                  linewidth=line_w,
                  err_kws={"color": "k", 'linewidth': 0.5},
                  capsize = 0.1,
                  markers='o',
                  palette=palette,
                  dodge=True, 
                  ax=ax12)
    
    ax12.get_legend().remove()
    ax12.set_xlabel("")
    ax12.set_xticks([0, 1, 2], ['','',''])
    
    ax12.set_yscale('log')
    ax12.set_ylim(0.15, 1.1)

    ax12.set_yticks([0.5, 1])  # Explicitly set desired ticks
    ax12.yaxis.set_major_formatter(ScalarFormatter())  # Force normal numbers
    ax12.yaxis.set_minor_locator(plt.NullLocator())  # Remove extra minor ticks
  
    #ax12.set_ylim(0, 1.5)
    #ax12.set_yticks(np.arange(0, 1.6, 0.5)) 
    #ax12.set_yticklabels([])
    ax12.axhline(y=1, c='k', linestyle='--')
    ax12.set_ylabel('')
    sns.despine(ax=ax12, top = True, right = True)
    
    df_long = df_e.melt(id_vars=['participant #', 'scale'], 
                        value_vars=['0.Line Length.0.rt', '2.Line Length.0.rt', '4.Line Length.0.rt'], 
                       var_name='Measure', 
                       value_name='Value')
    
    sns.pointplot(data=df_long, x='Measure', y='Value', 
                  hue='scale', 
                  hue_order=[30, 40, 60],
                  estimator=np.mean,
                  errorbar='se',
                  linewidth=line_w,
                  err_kws={"color": "k", 'linewidth': 0.5},
                  capsize = 0.1,
                  palette=palette,
                  dodge=True, markers='o',
                  ax=ax22)
    
    ax22.get_legend().remove()
    ax22.set_xlabel("External")
    ax22.set_xticks([0, 1, 2], ['T1','T2','T3'])
    ax22.set_ylim(1, 6)
    ax22.set_yticks(np.arange(1, 7, 1)) 
    ax22.set_yticklabels([])
    
    ax22.set_ylabel('')
    sns.despine(ax=ax22, top = True, right = True)
    
    
    df_w = df_stp.loc[df_stp['STP']=='Weak']
    df_long = df_w.melt(id_vars=['participant #', 'scale'], 
                       value_vars=['T1 ratio', 'T2 ratio', 'T3 ratio'], 
                       var_name='Measure', 
                       value_name='Value')
    
    sns.pointplot(data=df_long, x='Measure', y='Value', 
                  hue='scale', 
                  hue_order=[30,40, 60],
                  estimator=np.mean,
                  errorbar='se',
                  linewidth=line_w,
                  err_kws={"color": "k", 'linewidth': 0.5},
                  capsize = 0.1,
                  markers='o',
                  palette=palette,
                  dodge=True, 
                  ax=ax13)
    
    ax13.get_legend().remove()
    ax13.set_xlabel("")
    ax13.set_xticks([0, 1, 2], ['','',''])
    
    ax13.set_yscale('log')
    ax13.set_ylim(0.15, 1.1)
    ax13.set_yticks([0.5, 1])  # Explicitly set desired ticks
    ax13.yaxis.set_major_formatter(ScalarFormatter())  # Force normal numbers
    ax13.yaxis.set_minor_locator(plt.NullLocator())  # Remove extra minor ticks
  
    #ax13.set_yticks(np.arange(0, 1.6, 0.5)) 
    #ax13.set_yticklabels([])
    ax13.axhline(y=1, c='k', linestyle='--')
    ax13.set_ylabel('')
    sns.despine(ax=ax13, top = True, right = True)
    
    
    df_long = df_w.melt(id_vars=['participant #', 'scale'], 
                        value_vars=['0.Line Length.0.rt', '2.Line Length.0.rt', '4.Line Length.0.rt'], 
                       var_name='Measure', 
                       value_name='Value')
    
    sns.pointplot(data=df_long, x='Measure', y='Value', 
                  hue='scale', 
                  hue_order=[30,40, 60],
                  estimator=np.mean,
                  errorbar='se',
                  linewidth=line_w,
                  err_kws={"color": "k", 'linewidth': 0.5},
                  capsize = 0.1,
                  palette=palette,
                  dodge=True, markers='o',
                  ax=ax23)
    
    ax23.get_legend().remove()
    ax23.set_xlabel("Weak")
    ax23.set_xticks([0, 1, 2], ['T1','T2','T3'])
    ax23.set_ylim(1, 6)
    ax23.set_yticks(np.arange(1, 7, 1))
    ax23.set_yticklabels([])
    ax23.set_ylabel('')
    sns.despine(ax=ax23, top = True, right = True)
    
    plt.tight_layout()
    plt.savefig(file, dpi=600)

main('Figures/Figure 4_means_logscale.png')

#by median with RT
def main_med(file):
    df = pd.read_csv('Data/Experiment 9 - preregistered/merged - filtered.csv')
    
    options = [2, 7, 3]
    df_stp = df[df['Condition'].isin(options)]
    #change ms units to s 
    df_stp['0.Line Length.0.rt'] = df_stp['0.Line Length.0.rt'] / 1000
    df_stp['2.Line Length.0.rt'] = df_stp['2.Line Length.0.rt'] / 1000
    df_stp['4.Line Length.0.rt'] = df_stp['4.Line Length.0.rt'] / 1000
    
    fig = plt.figure()
    gs = fig.add_gridspec(2,3)
    ax11 = fig.add_subplot(gs[0,0])
    ax12 = fig.add_subplot(gs[0,1])
    ax13 = fig.add_subplot(gs[0,2])
    
    ax21 = fig.add_subplot(gs[1,0])
    ax22 = fig.add_subplot(gs[1,1])
    ax23 = fig.add_subplot(gs[1,2])
    
    line_w = 2.5
    
    palette = {30:'lightgray',  
               40:'gray',
               60:'k'}
    
    df_i = df_stp.loc[df_stp['STP']=='Internal']
    df_long = df_i.melt(id_vars=['participant #', 'scale'], 
                       value_vars=['T1 ratio', 'T2 ratio', 'T3 ratio'], 
                       var_name='Measure', 
                       value_name='Value')
    
    #df_long['Value'] = np.log10(df_long['Value'])
    sns.pointplot(data=df_long, x='Measure', y='Value', 
                  hue='scale', 
                  hue_order=[30, 40, 60],
                  estimator=np.mean,
                  errorbar='se',
                  linewidth=line_w,
                  err_kws={"color": "k", 'linewidth': 0.5},
                  capsize = 0.1,
                  markers='o',
                  palette=palette,
                  dodge=True, 
                  ax=ax11)
    
    ax11.set_xlabel("")
    ax11.set_xticks([0, 1, 2], ['','',''])
    
    ax11.set_ylim(0, 1.2)
    ax11.set_yticks(np.arange(0, 1.3, 0.5))
    
    ax11.axhline(y=1, c='k', linestyle='--')
    ax11.set_ylabel("")
    ax11.legend(title='Scale', loc='lower right', 
               title_fontsize='small', 
               fontsize='x-small')
    
    label1 = 'Estimate to stimulus\n ratio [A.U.]'
    ax11.set_ylabel(label1)
    sns.despine(ax=ax11, top = True, right = True)
    
    df_long = df_i.melt(id_vars=['participant #', 'scale'], 
                        value_vars=['0.Line Length.0.rt', '2.Line Length.0.rt', '4.Line Length.0.rt'], 
                       var_name='Measure', 
                       value_name='Value')
    
    sns.pointplot(data=df_long, x='Measure', y='Value', 
                  hue='scale', 
                  hue_order=[30, 40, 60],
                  estimator=np.mean,
                  errorbar='se',
                  linewidth=line_w,
                  err_kws={"color": "k", 'linewidth': 0.5},
                  capsize = 0.1,
                  palette=palette,
                  dodge=True, markers='o',
                  ax=ax21)
    
    ax21.get_legend().remove()
    ax21.set_xlabel("Internal")
    ax21.set_xticks([0, 1, 2], ['T1','T2','T3'])    
    ax21.set_ylim(1, 6)
    ax21.set_yticks(np.arange(1, 7, 1))
    sns.despine(ax=ax21, top = True, right = True)
    label1 = 'Response time\n[s]'
    ax21.set_ylabel(label1)
    
    df_e = df_stp.loc[df_stp['STP']=='External']
    df_long = df_e.melt(id_vars=['participant #', 'scale'], 
                       value_vars=['T1 ratio', 'T2 ratio', 'T3 ratio'], 
                       var_name='Measure', 
                       value_name='Value')
    
    sns.pointplot(data=df_long, x='Measure', y='Value', 
                  hue='scale', 
                  hue_order=[30,40, 60],
                  estimator=np.mean,
                  errorbar='se',
                  linewidth=line_w,
                  err_kws={"color": "k", 'linewidth': 0.5},
                  capsize = 0.1,
                  markers='o',
                  palette=palette,
                  dodge=True, 
                  ax=ax12)
    
    ax12.get_legend().remove()
    ax12.set_xlabel("")
    ax12.set_xticks([0, 1, 2], ['','',''])
    

    ax12.set_ylim(0, 1.2)
    ax12.set_yticks(np.arange(0, 1.3, 0.5)) 
    ax12.set_yticklabels([])
    ax12.axhline(y=1, c='k', linestyle='--')
    ax12.set_ylabel('')
    sns.despine(ax=ax12, top = True, right = True)
    
    df_long = df_e.melt(id_vars=['participant #', 'scale'], 
                        value_vars=['0.Line Length.0.rt', '2.Line Length.0.rt', '4.Line Length.0.rt'], 
                       var_name='Measure', 
                       value_name='Value')
    
    sns.pointplot(data=df_long, x='Measure', y='Value', 
                  hue='scale', 
                  hue_order=[30, 40, 60],
                  estimator=np.mean,
                  errorbar='se',
                  linewidth=line_w,
                  err_kws={"color": "k", 'linewidth': 0.5},
                  capsize = 0.1,
                  palette=palette,
                  dodge=True, markers='o',
                  ax=ax22)
    
    ax22.get_legend().remove()
    ax22.set_xlabel("External")
    ax22.set_xticks([0, 1, 2], ['T1','T2','T3'])
    ax22.set_ylim(1, 6)
    ax22.set_yticks(np.arange(1, 7, 1)) 
    ax22.set_yticklabels([])
    
    ax22.set_ylabel('')
    sns.despine(ax=ax22, top = True, right = True)
    
    
    df_w = df_stp.loc[df_stp['STP']=='Weak']
    df_long = df_w.melt(id_vars=['participant #', 'scale'], 
                       value_vars=['T1 ratio', 'T2 ratio', 'T3 ratio'], 
                       var_name='Measure', 
                       value_name='Value')
    
    sns.pointplot(data=df_long, x='Measure', y='Value', 
                  hue='scale', 
                  hue_order=[30,40, 60],
                  estimator=np.mean,
                  errorbar='se',
                  linewidth=line_w,
                  err_kws={"color": "k", 'linewidth': 0.5},
                  capsize = 0.1,
                  markers='o',
                  palette=palette,
                  dodge=True, 
                  ax=ax13)
    
    ax13.get_legend().remove()
    ax13.set_xlabel("")
    ax13.set_xticks([0, 1, 2], ['','',''])
     
    ax13.set_ylim(0, 1.2)
    ax13.set_yticks(np.arange(0, 1.3, 0.5)) 
    ax13.set_yticklabels([])
    ax13.axhline(y=1, c='k', linestyle='--')
    ax13.set_ylabel('')
    sns.despine(ax=ax13, top = True, right = True)
    
    
    df_long = df_w.melt(id_vars=['participant #', 'scale'], 
                        value_vars=['0.Line Length.0.rt', '2.Line Length.0.rt', '4.Line Length.0.rt'], 
                       var_name='Measure', 
                       value_name='Value')
    
    sns.pointplot(data=df_long, x='Measure', y='Value', 
                  hue='scale', 
                  hue_order=[30,40, 60],
                  estimator=np.mean,
                  errorbar='se',
                  linewidth=line_w,
                  err_kws={"color": "k", 'linewidth': 0.5},
                  capsize = 0.1,
                  palette=palette,
                  dodge=True, markers='o',
                  ax=ax23)
    
    ax23.get_legend().remove()
    ax23.set_xlabel("Weak")
    ax23.set_xticks([0, 1, 2], ['T1','T2','T3'])
    ax23.set_ylim(1, 6)
    ax23.set_yticks(np.arange(1, 7, 1))
    ax23.set_yticklabels([])
    ax23.set_ylabel('')
    sns.despine(ax=ax23, top = True, right = True)
    
    plt.tight_layout()
    plt.savefig(file, dpi=600)

main_med('Figures/Figure 4_medians.png')


#by median with fewshot
df = pd.read_csv('Data/Experiment 9 - preregistered/merged - filtered.csv')
options = [2, 7, 3]
df_stp = df[df['Condition'].isin(options)]
df_stp['few-shot'] = (df_stp["4.Line Length.0.answer"] - df_stp["0.Line Length.0.answer"])*100 / df_stp["0.Line Length.0.answer"]

fig = plt.figure()
gs = fig.add_gridspec(1,3)
ax11 = fig.add_subplot(gs[0,0])
ax12 = fig.add_subplot(gs[0,1])
ax13 = fig.add_subplot(gs[0,2])

line_w = 2.5

palette = {30:'lightgray',  
           40:'gray',
           60:'k'}

df_i = df_stp.loc[df_stp['STP']=='Internal']
sns.pointplot(data=df_i, x='scale', y='few-shot', 
              estimator=np.mean,
              errorbar='se',
              linewidth=line_w,
              err_kws={"color": "k", 'linewidth': 0.5},
              capsize = 0.1,
              #palette=palette,
              dodge=True, markers='o',
              ax=ax11)

#ax11.get_legend().remove()
ax11.set_xlabel("Internal")
#ax21.set_xticks([0, 1, 2], ['T1','T2','T3'])    
ax11.set_ylim(-100, 400)
ax11.set_yticks(np.arange(-100, 500, 100))
sns.despine(ax=ax11, top = True, right = True)


df_e = df_stp.loc[df_stp['STP']=='External']
sns.pointplot(data=df_e, x='scale', y='few-shot', 
              estimator=np.mean,
              errorbar='se',
              linewidth=line_w,
              err_kws={"color": "k", 'linewidth': 0.5},
              capsize = 0.1,
              #palette=palette,
              dodge=True, markers='o',
              ax=ax12)

#ax12.get_legend().remove()
ax12.set_xlabel("External")
#ax21.set_xticks([0, 1, 2], ['T1','T2','T3'])    
ax12.set_ylim(-100, 400)
ax12.set_yticks(np.arange(-100, 500, 100))
sns.despine(ax=ax12, top = True, right = True)



df_w = df_stp.loc[df_stp['STP']=='Weak']
sns.pointplot(data=df_w, x='scale', y='few-shot', 
              estimator=np.mean,
              errorbar='se',
              linewidth=line_w,
              err_kws={"color": "k", 'linewidth': 0.5},
              capsize = 0.1,
              #palette=palette,
              dodge=True, markers='o',
              ax=ax13)

#ax13.get_legend().remove()
ax13.set_xlabel("Weak")
#ax21.set_xticks([0, 1, 2], ['T1','T2','T3'])    
ax13.set_ylim(-100, 400)
ax13.set_yticks(np.arange(-100, 500, 100))
sns.despine(ax=ax13, top = True, right = True)

plt.tight_layout()

plt.savefig('Figures/Figure 4_few.png', dpi=600)


#preregistration: learning of ex>internal
x = df_stp.loc[(df_stp["STP"]=='External') ,"few-shot"]
y = df_stp.loc[(df_stp["STP"]=='Internal') ,"few-shot"]

stat = ttest(x, y, alternative='greater')
t = stat['T'][0]
p = stat['p-val'][0]
BF10 = stat['BF10'][0]
print(f"one-tailed t-test:\np = {np.round(p,decimals=5)}, t = {t:.2f}, BF10 = {BF10}")
#preregistered hypothesis (Internal<External) is n.s.

#x = df_stp.loc[(df_stp["STP"]=='Weak') ,"few-shot"]
x = df_stp.loc[(df_stp["STP"]=='External'),"T3 ratio"]
y = df_stp.loc[(df_stp["STP"]=='Internal'),"T3 ratio"]

print("Pooled data:")
print(f"- External: x͂ = {np.median(x):.2f}, x̄ = {np.mean(x):.2f} ± {np.std(x, ddof=1) / np.sqrt(len(x)):.2f}")
print(f"- Internal: x͂ = {np.median(y):.2f}, x̄ = {np.mean(y):.2f} ± {np.std(y, ddof=1) / np.sqrt(len(y)):.2f}")

stat = ttest(x, y, alternative='less')
t = stat['T'][0]
p = stat['p-val'][0]
BF10 = stat['BF10'][0]
print(f"-- t-test: p = {np.round(p,decimals=5)}, t = {t:.2f}, BF10 = {BF10}")

res = permutation_test((x, y), avg_statistic, 
                       vectorized=False,
                       n_resamples = 5000, 
                       alternative='less')

print(f"-- means diff {res.statistic:.2f}, p(prem.) = {res.pvalue:.4f}")


mw, p = mannwhitneyu(x,y, alternative='less')
print(f"-- MW test: p = {p:.5f}")

res = permutation_test((x, y), med_statistic, 
                       vectorized=False,
                       n_resamples = 5000, 
                       alternative='less')

print(f"-- medians diff {res.statistic:.2f}, p(prem.) = {res.pvalue:.4f}")

x = df_stp.loc[(df_stp["STP"]=='External') & (df_stp["scale"]==30) ,"T3 ratio"]
y = df_stp.loc[(df_stp["STP"]=='Internal') & (df_stp["scale"]==30),"T3 ratio"]

print("Fixed 30:")
print(f"- External: x͂ = {np.median(x):.2f}, x̄ = {np.mean(x):.2f} ± {np.std(x, ddof=1) / np.sqrt(len(x)):.2f}")
print(f"- Internal: x͂ = {np.median(y):.2f}, x̄ = {np.mean(y):.2f} ± {np.std(y, ddof=1) / np.sqrt(len(y)):.2f}")

stat = ttest(x, y, alternative='less')
t = stat['T'][0]
p = stat['p-val'][0]
BF10 = stat['BF10'][0]
print(f"-- t-test: p = {np.round(p,decimals=5)}, t = {t:.2f}, BF10 = {BF10}")

res = permutation_test((x, y), avg_statistic, 
                       vectorized=False,
                       n_resamples = 5000, 
                       alternative='less')

print(f"-- means diff {res.statistic:.2f}, p(prem.) = {res.pvalue:.4f}")


mw, p = mannwhitneyu(x,y, alternative='less')
print(f"-- MW test: p = {p:.5f}")

res = permutation_test((x, y), med_statistic, 
                       vectorized=False,
                       n_resamples = 5000, 
                       alternative='less')

print(f"-- medians diff {res.statistic:.2f}, p(prem.) = {res.pvalue:.4f}")

'''
options = [1, 6, 2, 7, 3]
df_stp = df[df['Condition'].isin(options)]


plt.scatter(df_stp['scale'].loc[df_stp['STP']=='Internal'],
            df_stp['0.Line Length.0.answer'].loc[df_stp['STP']=='Internal'],
            facecolor='none', 
            edgecolor='b',
            label='Internal')

x , y = df_stp['scale'].loc[df_stp['STP']=='Internal'], df_stp['0.Line Length.0.answer'].loc[df_stp['STP']=='Internal'],

m, b = np.polyfit(x, y, 1)
slope, intercept, r_value, p_value, std_err = linregress(x, y)
print(f'Internal:\n-intercept:{intercept:.3f}, slope: {slope:.3f}\n-R square linear: {r_value**2:.2f}, \n-p value: {p_value} ')
plt.plot(x, m*x+b, c='b')

plt.scatter(df_stp['scale'].loc[df_stp['STP']=='External'],
            df_stp['0.Line Length.0.answer'].loc[df_stp['STP']=='External'],
            facecolor='none', 
            edgecolor='r',
            label='External')

x , y = df_stp['scale'].loc[df_stp['STP']=='External'], df_stp['0.Line Length.0.answer'].loc[df_stp['STP']=='External'],

m, b = np.polyfit(x, y, 1)
slope, intercept, r_value, p_value, std_err = linregress(x, y)
print(f'External:\n-intercept:{intercept:.3f}, slope: {slope:.3f}\n-R square linear: {r_value**2:.2f}, \n-p value: {p_value} ')
plt.plot(x, m*x+b, c='r')

plt.plot(np.arange(10, 70), np.arange(10, 70), c='k', linestyle='--' )    
plt.legend()
plt.show()

X = sm.add_constant(x)
mod = sm.OLS(y, X)
res = mod.fit()
print (res.conf_int(0.05))   # 95% confidence interval

'''