# -*- coding: utf-8 -*-
"""
Groups analysis - coherent prior, weak prior, and illusory prior 
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.anova import AnovaRM
from scipy.stats import linregress, sem, ks_2samp, wilcoxon, bartlett, fligner, levene, f_oneway, norm, ttest_ind, ttest_1samp, ttest_ind_from_stats, permutation_test, spearmanr, mannwhitneyu, shapiro
from pingouin import bayesfactor_ttest, welch_anova, ttest

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter, FixedLocator
import seaborn as sns
sns.set_style("whitegrid")

def med_statistic(x, y):
    return np.median(x) - np.median(y)

def avg_statistic(x, y):
    return np.mean(x) - np.mean(y)

def spearmanr_statistic(x, y):
    return spearmanr(x, y)[0]

df = pd.read_csv('Data/Experiment 9 - preregistered/merged - filtered.csv')

conditions_names = {0: 'Pooled',
                    1:'F-15',
                    2: 'F-30',
                    3: 'F-60',
                    4: 'A-up [20-40-20]',
                    5: 'A-down [40-20-40]',
                    6: 'F-20',
                    7: 'F-40',
                    8: 'A-down [15-10-15]'}

def plot_prior(df, file): 
    
    palette = {'External': 'lightsteelblue', 
               'Internal': "#1f77b4", 
               'Weak': 'dodgerblue'}
    
    line_w = 2.5

    fig = plt.figure()
    gs = fig.add_gridspec(2,2)
    ax11 = fig.add_subplot(gs[0,0])
    ax12 = fig.add_subplot(gs[1,0])
    ax13 = fig.add_subplot(gs[1,1])
    ax14 = fig.add_subplot(gs[0,1])

    sns.countplot(data=df, x="STP",
                 edgecolor='k',
                 order=['Weak', 'Internal', 'External'],
                 palette=palette,
                 ax=ax11)

    ax11.set_xlabel("")
    ax11.set_xticks([0, 1, 2], ['','', ''])
    ax11.set_ylim(0, 120)
    ax11.set_yticks(np.arange(0, 121, 30))
    ax11.set_ylabel("Participants\n[count]")
    ax11.text(-0.25, 1.05, '(a)', 
            transform=ax11.transAxes,
            fontsize=12, 
            fontweight='bold', 
            va='top', 
            ha='right')

    sns.despine(ax=ax11, top = True, right = True)
    
    sns.boxplot(
        x="STP", y="3.Recent Prior Distorted.0.rt", data=df,
        palette=palette, 
        order=['Weak', 'Internal', 'External'],
        ax=ax12)
    
    ax12.set_xticks([0, 1, 2], ["I don't see\nanything",'I see a face\nor whole\nperson', 'I see a part\nof a\nperson'])
    ax12.set_xticklabels(ax12.get_xticklabels(), fontsize=7.5)
    ax12.set_ylim(0, 6)
    ax12.set_yticks(np.arange(0, 7, 1))
    ax12.set_xlabel("")
    ax12.set_ylabel('Response time\n[s]')
    ax12.text(-0.25, 1.05, '(b)', 
            transform=ax12.transAxes,
            fontsize=12, 
            fontweight='bold', 
            va='top', 
            ha='right')
    sns.despine(ax=ax12, top = True, right = True)
    
    sns.boxplot(
        x="STP", y="two-shot", hue="STP",
        data=df,
        palette=palette,
        medianprops={'color': 'black'},
        fliersize=3,
        legend=False,
        order=['Weak', 'Internal', 'External'],
        ax=ax13
    )
    
    ax13.set_xticks([0, 1, 2], ['','', ''])
    ax13.set_ylabel('Change [%]\n(' + r'$\frac{T3 - T1}{T1}$' + ')')
    ax13.set_ylim(-100, 600)
    ax13.set_xlabel('')
    ax13.set_yticks(np.arange(-100, 601, 100))
    ax13.axhline(0, color='k')
    sns.despine(ax=ax13)
    ax13.text(-0.3, 1.05, '(d)', 
            transform=ax13.transAxes,
            fontsize=12, 
            fontweight='bold', 
            va='top', 
            ha='right')

    df_long = df.melt(id_vars=['participant #', 'STP'], 
                       value_vars=['T1 ratio', 'T2 ratio', 'T3 ratio'], 
                       var_name='Measure', 
                       value_name='Value')
    
    sns.pointplot(data=df_long, x='Measure', y='Value', 
                  hue='STP', 
                  hue_order=['Weak', 'Internal', 'External'],
                  estimator=np.median,
                  errorbar='se',
                  linewidth=line_w,
                  err_kws={"color": "k", 'linewidth': 0.5},
                  capsize = 0.1,
                  markers='o',
                  palette=palette,
                  alpha=1.0,
                  dodge=True,
                  ax=ax14)

    ax14.get_legend().remove()
    ax14.set_xticks([0, 1, 2], ['T1','T2','T3'])
    
    yticks = np.round(np.arange(0, 1.3, 0.2),1)
    ytick_labels = ['0' if y == 0 else f'{y:.1f}' for y in yticks]
    ax14.set_ylim(0, 1.2)
    ax14.set_yticks(yticks)                   # Set tick positions
    ax14.set_yticklabels(ytick_labels)        # Set corresponding labels

    ax14.axhline(y=1, c='k', linestyle='--')
    ax14.set_xlabel('Trials')
    label1 = 'Perceived\nlength ratio\n[A.U.]'
    ax14.set_ylabel(label1)
    sns.despine(ax=ax14, top = True, right = True)
    ax14.text(-0.3, 1.05, '(c)', 
            transform=ax14.transAxes,
            fontsize=12, 
            fontweight='bold', 
            va='top', 
            ha='right')
    
    
    plt.tight_layout()
    plt.savefig(file, dpi=600)
    
options = [2, 3, 7]
df_stp = df[df['Condition'].isin(options)]
df_stp['3.Recent Prior Distorted.0.rt'] = df_stp['3.Recent Prior Distorted.0.rt'] / 1000
df_stp['two-shot'] = (df_stp["4.Line Length.0.answer"] - df_stp["0.Line Length.0.answer"])*100 / df_stp["0.Line Length.0.answer"]

plot_prior(df_stp, f'Figures/Figure 4.png')

#stats:
##(1) preregistred hypothesis is n.s. (ex>internal)
x = df_stp.loc[(df_stp["STP"]=='External') ,"two-shot"]
y = df_stp.loc[(df_stp["STP"]=='Internal') ,"two-shot"]

stat = ttest(x, y, alternative='greater')
t = stat['T'][0]
p = stat['p-val'][0]
BF10 = stat['BF10'][0]
print(f"one-tailed t-test:\np = {np.round(p,decimals=5)}, t = {t:.2f}, BF10 = {BF10}")

##(2) updated hypothesis - the delta in ratio: int>ext 
x = df_stp.loc[(df_stp["STP"]=='External'),"T1 ratio"]
y = df_stp.loc[(df_stp["STP"]=='Internal'),"T1 ratio"]

print("Pooled data for T1 ratio")
print(f"- External: x͂ = {np.median(x):.2f}, x̄ = {np.mean(x):.2f} ± {np.std(x, ddof=1) / np.sqrt(len(x)):.2f}")
print(f"- Internal: x͂ = {np.median(y):.2f}, x̄ = {np.mean(y):.2f} ± {np.std(y, ddof=1) / np.sqrt(len(y)):.2f}")

stat = ttest(x, y, alternative='two-sided')
t = stat['T'][0]
p = stat['p-val'][0]
BF10 = stat['BF10'][0]
print(f"-- t-test: p = {np.round(p,decimals=5)}, t = {t:.2f}, BF10 = {BF10}")

df_stp["T3-T1 ratio"] = df_stp["T3 ratio"] - df_stp["T1 ratio"]
x = df_stp.loc[(df_stp["STP"]=='External'),"T3-T1 ratio"]
y = df_stp.loc[(df_stp["STP"]=='Internal'),"T3-T1 ratio"]

stat = ttest(x, y, alternative='two-sided')
t = stat['T'][0]
p = stat['p-val'][0]
BF10 = stat['BF10'][0]
print(f"-- t-test: p = {np.round(p,decimals=5)}, t = {t:.2f}, BF10 = {BF10}")


res = permutation_test((x, y), med_statistic, 
                       vectorized=False,
                       n_resamples = 1000, 
                       alternative='less')

print(f"-- means diff {res.statistic:.2f}, p(prem.) = {res.pvalue:.4f}")

mw, p = mannwhitneyu(x,y, alternative='less')
print(f"-- MW test: p = {p:.5f}")

x = df_stp.loc[(df_stp["STP"]=='External') & (df_stp["scale"]==30) ,"T3 ratio"]
y = df_stp.loc[(df_stp["STP"]=='Internal') & (df_stp["scale"]==30) ,"T3 ratio"]

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

###################### dradt code ######################
'''
analysis #1:
does the estimate ratio higher for internal group (and lower RTs?)

TL;DR - on avg yes, but small group size doesn't allow per condition analysis  
'''

def plot_prior(df, file): 
    
    palette = {'External': 'lightsteelblue', 
               'Internal': "#1f77b4", 
               'Weak': 'dodgerblue'}
    
    line_w = 2.5

    '''
    fig, ax5 = plt.subplots(2,1)
        
    sns.countplot(data=df, x="STP",
                 edgecolor='k',
                 order=['Weak', 'Internal', 'External'],
                 palette=palette,
                 #shrink=0.8,
                 ax=ax5[0])

    ax5[0].set_xticks([0, 1, 2], ['I dont see \nanything','I see a\nface / whole person', 'I see a\npart of a person'])
    ax5[0].set_xlabel("")

    ax5[0].set_ylim(0, 120)
    ax5[0].set_yticks(np.arange(0, 121, 30))
    ax5[0].set_ylabel("Participants\n[#]")

    sns.despine(ax=ax5[0], top = True, right = True)
    
    sns.boxplot(
        x="STP", y="3.Recent Prior Distorted.0.rt", data=df,
        palette=palette, 
        order=['Weak', 'Internal', 'External'],
        ax=ax5[1])

    ax5[1].set_ylim(1, 6)
    ax5[1].set_yticks(np.arange(1, 7, 1))
    ax5[1].set_xlabel("")
    ax5[1].set_ylabel('Response time\n[s]')
    sns.despine(ax=ax5[1], top = True, right = True)
    #ax5[1].set_xticks([0, 1, 2], ['','', ''])

    
    plt.tight_layout()
    plt.savefig('Figures/Figure s1_pooled.png', dpi=600)
    '''
    fig = plt.figure()
    gs = fig.add_gridspec(2,4)
    ax11 = fig.add_subplot(gs[0,1])
    ax12 = fig.add_subplot(gs[0,2])
    ax13 = fig.add_subplot(gs[0,3])
    ax14 = fig.add_subplot(gs[0,0])
    
    ax21 = fig.add_subplot(gs[1,1])
    ax22 = fig.add_subplot(gs[1,2])
    ax23 = fig.add_subplot(gs[1,3])
    ax24 = fig.add_subplot(gs[1,0])
    
    df_1 = df.loc[df['Condition']==2]
    df_long = df_1.melt(id_vars=['participant #', 'STP'], 
                       value_vars=['T1 ratio', 'T2 ratio', 'T3 ratio'], 
                       var_name='Measure', 
                       value_name='Value')
    
    sns.pointplot(data=df_long, x='Measure', y='Value',
                  hue='STP', 
                  hue_order=['Weak', 'Internal', 'External'],
                  estimator=np.median,
                  errorbar='se',
                  linewidth=line_w,
                  err_kws={"color": "k", 'linewidth': 0.5},
                  capsize = 0.1,
                  markers='o',
                  palette=palette,
                  alpha=1.0,
                  dodge=True, 
                  ax=ax11)
    
    ax11.get_legend().remove()
    ax11.set_xlabel("")
    ax11.set_xticks([0, 1, 2], ['','',''])

    #ax11.set_xticks([0, 1, 2], ['T1','T2','T3'])
    ax11.set_ylim(0, 1.5)
    ax11.set_yticks(np.arange(0, 1.6, 0.5))   
    ax11.set_yticklabels([])
    
    ax11.axhline(y=1, c='k', linestyle='--')
    ax11.set_ylabel("")

    sns.despine(ax=ax11, top = True, right = True)
    
    df_2 = df.loc[df['Condition']==7]
    df_long = df_2.melt(id_vars=['participant #', 'STP'], 
                       value_vars=['T1 ratio', 'T2 ratio', 'T3 ratio'], 
                       var_name='Measure', 
                       value_name='Value')
    
    sns.pointplot(data=df_long, x='Measure', y='Value', 
                  hue='STP', 
                  hue_order=['Weak', 'Internal', 'External'],
                  estimator=np.median,
                  errorbar='se',
                  linewidth=line_w,
                  err_kws={"color": "k", 'linewidth': 0.5},
                  capsize = 0.1,
                  markers='o',
                  palette=palette,
                  alpha=1.0,

                  #fliersize=3,
                  dodge=True, 
                  ax=ax12)
    
    ax12.get_legend().remove()
    ax12.set_xlabel("")
    ax12.set_xticks([0, 1, 2], ['','',''])
    #ax12.set_xticks([0, 1, 2], ['T1','T2','T3'])
    ax12.set_ylim(0, 1.5)
    ax12.set_yticks(np.arange(0, 1.6, 0.5)) 
    ax12.set_yticklabels([])

    ax12.axhline(y=1, c='k', linestyle='--')
    ax12.set_ylabel('')
    sns.despine(ax=ax12, top = True, right = True)
    
    df_3 = df.loc[df['Condition']==3]
    df_long = df_3.melt(id_vars=['participant #', 'STP'], 
                       value_vars=['T1 ratio', 'T2 ratio', 'T3 ratio'], 
                       var_name='Measure', 
                       value_name='Value')
    
    sns.pointplot(data=df_long, x='Measure', y='Value', 
                  hue='STP', 
                  hue_order=['Weak', 'Internal', 'External'],
                  estimator=np.median,
                  errorbar='se',
                  linewidth=line_w,
                  err_kws={"color": "k", 'linewidth': 0.5},
                  capsize = 0.1,
                  markers='o',
                  palette=palette,
                  alpha=1.0,

                  #fliersize=3,
                  dodge=True, 
                  ax=ax13)

    ax13.get_legend().remove()
    ax13.set_xlabel("")
    #ax13.set_xticks([0, 1, 2], ['T1','T2','T3'])
    ax13.set_xticks([0, 1, 2], ['','',''])
    ax13.set_ylim(0, 1.5)
    ax13.set_yticks(np.arange(0, 1.6, 0.5)) 
    ax13.set_yticklabels([])

    ax13.axhline(y=1, c='k', linestyle='--')
    ax13.set_ylabel('')
    sns.despine(ax=ax13, top = True, right = True)
    
    df_long = df.melt(id_vars=['participant #', 'STP'], 
                       value_vars=['T1 ratio', 'T2 ratio', 'T3 ratio'], 
                       var_name='Measure', 
                       value_name='Value')
    
    sns.pointplot(data=df_long, x='Measure', y='Value', 
                  hue='STP', 
                  hue_order=['Weak', 'Internal', 'External'],
                  estimator=np.median,
                  errorbar='se',
                  linewidth=line_w,
                  err_kws={"color": "k", 'linewidth': 0.5},
                  capsize = 0.1,
                  markers='o',
                  palette=palette,
                  alpha=1.0,

                  #fliersize=3,
                  dodge=True,
                  ax=ax14)

    ax14.get_legend().remove()
    #ax14.legend(title='Prior', loc='upper left', 
    #           title_fontsize='x-small', 
    #           fontsize='xx-small')
    
    #ax14.set_xticks([0, 1, 2], ['T1','T2','T3'])
    ax14.set_xticks([0, 1, 2], ['','',''])
    ax14.set_ylim(0, 1.5)
    ax14.set_yticks(np.arange(0, 1.6, 0.5)) 
    ax14.axhline(y=1, c='k', linestyle='--')
    ax14.set_xlabel('')
    label1 = 'Estimate to stimulus\n ratio [A.U.]'
    ax14.set_ylabel(label1)
    
    sns.despine(ax=ax14, top = True, right = True)
    
    
    df_1 = df.loc[df['Condition']==2]
    df_long = df_1.melt(id_vars=['participant #', 'STP'], 
                       value_vars=['0.Line Length.0.rt', '2.Line Length.0.rt', '4.Line Length.0.rt'], 
                       var_name='Measure', 
                       value_name='Value')
    
    sns.pointplot(data=df_long, x='Measure', y='Value', 
                  hue='STP', 
                  hue_order=['Weak', 'Internal', 'External'],
                  estimator=np.median,
                  errorbar='se',
                  linewidth=line_w,
                  err_kws={"color": "k", 'linewidth': 0.5},
                  capsize = 0.1,
                  palette=palette,
                  alpha=1.0,

                  dodge=True, markers='o',
                  ax=ax21)
    
    ax21.get_legend().remove()
    ax21.set_xlabel("Fixed 30")
    ax21.set_xticks([0, 1, 2], ['T1','T2','T3'])    
    ax21.set_ylim(1, 6)
    ax21.set_yticks(np.arange(1, 7, 1))
    ax21.set_yticklabels([])

    ax21.set_ylabel('')
    sns.despine(ax=ax21, top = True, right = True)
    
    df_2 = df.loc[df['Condition']==7]
    df_long = df_2.melt(id_vars=['participant #', 'STP'], 
                       value_vars=['0.Line Length.0.rt', '2.Line Length.0.rt', '4.Line Length.0.rt'], 
                       var_name='Measure', 
                       value_name='Value')
    
    sns.pointplot(data=df_long, x='Measure', y='Value', 
                  hue='STP', 
                  hue_order=['Weak', 'Internal', 'External'],
                  estimator=np.median,
                  errorbar='se',
                  linewidth=line_w,
                  err_kws={"color": "k", 'linewidth': 0.5},
                  capsize = 0.1,
                  palette=palette,
                  alpha=1.0,

                  dodge=True, markers='o',
                  ax=ax22)
    
    ax22.get_legend().remove()
    ax22.set_xlabel("Fixed 40")
    ax22.set_xticks([0, 1, 2], ['T1','T2','T3'])
    ax22.set_ylim(1, 6)
    ax22.set_yticks(np.arange(1, 7, 1)) 
    ax22.set_yticklabels([])

    ax22.set_ylabel('')
    sns.despine(ax=ax22, top = True, right = True)
    
    df_3 = df.loc[df['Condition']==3]
    df_long = df_3.melt(id_vars=['participant #', 'STP'], 
                       value_vars=['0.Line Length.0.rt', '2.Line Length.0.rt', '4.Line Length.0.rt'], 
                       var_name='Measure', 
                       value_name='Value')
    
    sns.pointplot(data=df_long, x='Measure', y='Value', 
                  hue='STP', 
                  hue_order=['Weak', 'Internal', 'External'],
                  estimator=np.median,
                  errorbar='se',
                  linewidth=line_w,
                  err_kws={"color": "k", 'linewidth': 0.5},
                  capsize = 0.1,
                  palette=palette,
                  alpha=1.0,

                  dodge=True, markers='o',
                  ax=ax23)
    
    ax23.get_legend().remove()
    ax23.set_xlabel("Fixed 60")
    ax23.set_xticks([0, 1, 2], ['T1','T2','T3'])
    ax23.set_ylim(1, 5)
    ax23.set_yticks(np.arange(1, 7, 1))
    ax23.set_yticklabels([])

    ax23.set_ylabel('')
    sns.despine(ax=ax23, top = True, right = True)
    
    df_long = df.melt(id_vars=['participant #', 'STP'], 
                       value_vars=['0.Line Length.0.rt', '2.Line Length.0.rt', '4.Line Length.0.rt'], 
                       var_name='Measure', 
                       value_name='Value')
    
    sns.pointplot(data=df_long, x='Measure', y='Value', 
                  hue='STP', 
                  hue_order=['Weak', 'Internal', 'External'],
                  estimator=np.median,
                  errorbar='se',
                  linewidth=line_w,
                  err_kws={"color": "k", 'linewidth': 0.5},
                  capsize = 0.1,
                  markers='o',
                  palette=palette,
                  alpha=1.0,

                  #fliersize=3,
                  dodge=True,
                  ax=ax24)

    ax24.set_xlabel('Pooled')
    ax24.get_legend().remove()

    label1 = 'Response time\n[s]'
    ax24.set_ylabel(label1)
    ax24.set_xticks([0, 1, 2], ['T1','T2','T3'])
    ax24.set_ylim(1, 5)
    ax24.set_yticks(np.arange(1, 7, 1))
    sns.despine(ax=ax24, top = True, right = True)

    plt.tight_layout()
    plt.show()
    #plt.savefig(file, dpi=600)
    
options = [2, 3, 7]
df_stp = df[df['Condition'].isin(options)]
df_stp['0.Line Length.0.rt'] = df_stp['0.Line Length.0.rt'] / 1000
df_stp['2.Line Length.0.rt'] = df_stp['2.Line Length.0.rt'] / 1000
df_stp['4.Line Length.0.rt'] = df_stp['4.Line Length.0.rt'] / 1000
df_stp['3.Recent Prior Distorted.0.rt'] = df_stp['3.Recent Prior Distorted.0.rt'] / 1000
df_stp['1.Recent Prior Normal.0.rt'] = df_stp['1.Recent Prior Normal.0.rt'] / 1000
df_stp['two-shot'] = (df_stp["4.Line Length.0.answer"] - df_stp["0.Line Length.0.answer"])*100 / df_stp["0.Line Length.0.answer"]

plot_prior(df_stp, f'Figures/Figure s1_all.png')


'''
analysis #2:
when sensory noise increases (larger scale), 
does the interal group rely more on prior? 

TL;DR - not enoght participants in this group... 
'''
#option 1 - scales per prior group
options = [2, 7, 3]
df_stp = df[df['Condition'].isin(options)]
df_stp['two-shot'] = (df_stp["4.Line Length.0.answer"] - df_stp["0.Line Length.0.answer"])*100 / df_stp["0.Line Length.0.answer"]

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
sns.pointplot(data=df_i, x='scale', y='two-shot', 
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
sns.pointplot(data=df_e, x='scale', y='two-shot', 
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
sns.pointplot(data=df_w, x='scale', y='two-shot', 
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
plt.show()

plt.savefig('Figures/Figure 4_few.png', dpi=600)


#option 2 - estimate ratio of the three trials per group
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
    plt.show()
    #plt.savefig(file, dpi=600)

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

