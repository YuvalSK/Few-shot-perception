# -*- coding: utf-8 -*-
"""
Few-shot learning of visual length
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

from statsmodels.stats.anova import AnovaRM
from scipy.stats import sem, ks_2samp, wilcoxon, bartlett, fligner, levene, f_oneway, norm, ttest_ind, ttest_1samp, ttest_ind_from_stats, permutation_test, spearmanr, mannwhitneyu, shapiro
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
analysis by scale, to test individual change
'''

df = pd.read_csv('Data/Experiment 9 - preregistered/merged - filtered.csv')
fig, ax = plt.subplots(2,1)

options = [1, 2, 3, 6, 7]
df_stp = df[df['Condition'].isin(options)]
df_stp['few-shot'] = (df_stp["4.Line Length.0.answer"] - df_stp["0.Line Length.0.answer"])*100 / df_stp["0.Line Length.0.answer"]

palette = {'15':'lightgray', '20':'darkgray', 
           '30':'gray', '40':'dimgrey',
           '60':'dimgray'}

sns.boxplot(
    x="scale", y="few-shot", 
    data=df_stp,
    medianprops={'color':'black'},
    fliersize=3,
    palette=palette, 
    order=[15, 20, 30, 40, 60],
    ax=ax[1])

ax[1].set_yticks(np.arange(-100, 601, 100))
ax[1].set_ylim(-100, 600)
label = 'Change [%]\n(' + r'$\frac{T3 - T1}{T1}$' + ')'
ax[1].set_ylabel(label)

handles, labels = ax[0].get_legend_handles_labels()

ax[1].set_xticks(np.arange(0, 5, 1), ['Fixed 15', 'Fixed 20', 'Fixed 30', 'Fixed 40', 'Fixed 60'])
ax[1].set_xlabel("")
ax[1].axhline(y=0, c='k', linestyle='-')
sns.despine(ax=ax[1], top = True, right = True)

'''
analysis by prev stimulus direction 
'''

df = pd.read_csv('Data/Experiment 9 - preregistered/merged - filtered.csv')
options = [6, 7, 4, 5] #without scales 15 and 20 (smaller learning effect)
df_stp = df[df['Condition'].isin(options)]

#calculating the % change as another measure of individual learning
df_stp['few-shot'] = (df_stp["4.Line Length.0.answer"] - df_stp["0.Line Length.0.answer"])*100 / df_stp["0.Line Length.0.answer"]

def modify_prev(row):
    if row['prev'] == 'Fixed':
        return f'Fixed {row["scale"]}'
    return row['prev']

df_stp['category'] = df_stp['prev']
df_stp['category'] = df_stp.apply(modify_prev, axis=1)

palette = {'Up':'green',
           'Fixed 20':'darkgray', 
           'Fixed 40':'dimgrey',
           'Down':'red'}

sns.boxplot(
    x="category", y="few-shot", 
    data=df_stp,
    medianprops={'color':'black'},
    fliersize=3,
    palette=palette, 
    order=['Up', 'Fixed 20', 'Fixed 40', 'Down'],
    ax=ax[0])

label = 'Change [%]\n(' + r'$\frac{T3 - T1}{T1}$' + ')'
ax[0].set_ylabel(label)

#handles, labels = ax[0].get_legend_handles_labels()
ax[0].set_xlabel("")
ax[0].set_xticks(np.arange(0, 4, 1), ['High adapt\n[20, 40, 20]', 'Fixed 20', 'Fixed 40', 'Low adapt\n [40, 20, 40]'])
ax[0].set_yticks(np.arange(-100, 601, 100))
ax[0].set_ylim(-100, 600)

ax[0].axhline(y=0, c='k', linestyle='-')
sns.despine(ax=ax[0], top = True, right = True)

plt.tight_layout()
plt.savefig('Figures/Figure 3_v2.png', dpi=600)


#stats...
x1 = df_stp.loc[(df_stp["scale"]==15),   "few-shot"]
x2 = df_stp.loc[(df_stp["scale"]==20),   "few-shot"]
x3 = df_stp.loc[(df_stp["scale"]==30),   "few-shot"]
x4 = df_stp.loc[(df_stp["scale"]==40),   "few-shot"]
x5 = df_stp.loc[(df_stp["scale"]==60),   "few-shot"]
for name, data in zip(["15", "20", "30", "40", "60"], [x1, x2, x3, x4, x5]):
    print(f"scale: {name}")
    stat = ttest(data, 0, paired=False, alternative='greater')
    t = stat['T'][0]
    df = stat['dof'][0]
    p = stat['p-val'][0]
    BF10 = stat['BF10'][0]
    print(f"- one-tailed p = {np.round(p,decimals=5)}, t({df}) = {t}, BF10 = {BF10}")



x = df_stp.loc[(df_stp["prev"]=='Up'),   "L3-L2"]
y = df_stp.loc[(df_stp["prev"]=='Fixed'),"L3-L2"]
z = df_stp.loc[(df_stp["prev"]=='Down'), "L3-L2"]


stat, p = f_oneway(x,y,z, nan_policy='omit')
print(f"ANOVA: F = {stat}, p = {p}")

stat_l, pval_l = levene(x, y, z, center='median', nan_policy='omit')
print(f"Levene's test: {pval_l}")

w = welch_anova(df_stp, dv="L3-L2", between = "prev")
print(f"Welch correction ANOVA: {w['F'][0]:.2f} p-value:{w['p-unc'][0]}")

#stats...
x = df_stp.loc[(df_stp["prev"]=='Up'),   "few-shot"]
y = df_stp.loc[(df_stp["prev"]=='Fixed'),"few-shot"]
z = df_stp.loc[(df_stp["prev"]=='Down'), "few-shot"]

for name, data in zip(["Up", "Fixed", "Down"], [x, y, z]):
    print(f"{name}: \nx͂ = {np.median(data):.1f}, x̄ = {np.mean(data):.1f} ± {sem(data, ddof=1):.1f}")
    stat = ttest(data, 0, paired=False, alternative='greater')
    t = stat['T'][0]
    df = stat['dof'][0]
    p = stat['p-val'][0]
    BF10 = stat['BF10'][0]
    print(f"- one-tailed t-test: p = {np.round(p,decimals=5)}, t({df}) = {t:.1f}, BF10 = {BF10}")
    #w_stat, p = wilcoxon(data - 0, alternative='greater')  # One-tailed test
    #print(f"- Wilcoxon: W={w_stat:.3f}, p={p}")
    
stat, p = f_oneway(x,y,z)
print(f"ANOVA: F = {stat}, p = {p}")

stat_l, pval_l = levene(x, y, z, center='median')
print(f"Levene's test: {pval_l}")

stat_b, pval_b = bartlett(x, y, z)
print(f"Brown-Forsythe test: {stat_b} p-value:{pval_b}")

w = welch_anova(df_stp, dv="few-shot", between = "prev")
print(f"Welch correction ANOVA: {w['F'][0]:.2f} p-value:{w['p-unc'][0]}")

res = permutation_test((x, z), statistic, 
                       vectorized=False,
                       n_resamples = 9999, 
                       alternative='two-sided')

print(f"medinas: {np.median(x)}, External: {np.median(z)}")
print(f" diff {res.statistic}, p(prem.) = {res.pvalue:.4f}")

'''
#for mixed model with estimates in JASP:
df_long = df_stp.melt(id_vars=['participant #', 'STP', 'prev', 'scale'], 
                   value_vars=['0.Line Length.0.answer', '2.Line Length.0.answer', '4.Line Length.0.answer'], 
                   var_name='Trial', 
                   value_name='Estimate')
df_long.to_csv('Data/merged_long.csv')
'''
#adaptation plots
def adapt_plots(df_response, f):
    fig, ax = plt.subplots(2,2)
    l_max = 105
    leg_s = 7
    options = [6, 4]
    df_filtered = df_response[(df_response['condition'].isin(options))]
    df_filtered['RT'] = df_filtered['RT'] / 1000
      
    c = ['gray','green']
    c1 = 'seagreen' 
    l = ['Fixed 20','Adapt up']
      
    sns.boxplot(x='trial', y='response', 
                data=df_filtered, 
                hue='condition', hue_order=options,
                medianprops={'color':'black'},
                fliersize=3,
                #showmeans=True, 
                #meanprops={"marker":">","markerfacecolor":"orange"},
                #err_kws={"color": "gray", 'linewidth': 1.2},
                #capsize = 0.1,
                palette=c,
                ax=ax[0,0])
    sns.despine(ax=ax[0,0], top = True, right = True)
    ax[0,0].get_xaxis().set_visible(False)
    ax[0,0].set_ylabel('Length estimate\n[# of underbars]')        
    ax[0,0].set_ylim((0, l_max))
    ax[0,0].set_yticks(np.arange(0,l_max, 20))

    
    legend_elements = [
    Patch(facecolor=c[0], label=l[0]),
    Patch(facecolor=c[1], label=l[1])
    ]
    
    ax[0,0].legend(handles=legend_elements, loc='upper right', fontsize=leg_s)  

    sns.boxplot(x='trial', y='RT', 
                data=df_filtered, 
                hue='condition', hue_order=options,
                medianprops={'color':'black'}, 
                fliersize=3,

                #showmeans=True, 
                #meanprops={"marker":">", "markerfacecolor":"orange"},
                #errorbar='se',
                #err_kws={"color": "gray", 'linewidth': 1.2},
                #capsize = 0.1,
                palette=c,
                ax=ax[1,0])
    sns.despine(ax=ax[1,0], top = True, right = True)
    
    #ax[1,0].get_xaxis().set_visible(False)
    ax[1,0].set_ylabel('Response time\n[s]')
    
        
    #val = df_filtered.groupby(['trial','condition'])['RT'].mean()['T2']
    #for bar in ax[1,0].patches:
    #    if np.isclose(bar.get_height(), val[4], atol=0.01):
    #        bar.set_color(c1)
            
    ax[1,0].set_ylim((0, 8))
    ax[1,0].set_xlabel('')
    
    
    options = [7, 5]
    df_filtered = df_response[df_response['condition'].isin(options)]
    df_filtered['RT'] = df_filtered['RT'] / 1000

    c = ['gray','red']
    c1 = 'salmon' 
    l = ['Fixed 40','Adapt down']
    
    sns.boxplot(x='trial', y='response', 
                data=df_filtered, 
                hue='condition', hue_order=options,
                medianprops={'color':'black'}, 
                fliersize=3,

                #showmeans=True, 
                #meanprops={"marker":">","markeredgecolor":"orange"},
    
                #errorbar='se',
                #err_kws={"color": "gray", 'linewidth': 1.2},
                #capsize = 0.1,
                palette=c,
                ax=ax[0,1])
    
    #for i, box in enumerate(ax[0,1].artists):
    #    if i == 1:
    #        box.set_facecolor(c1)
        
    #val = df_filtered.groupby(['trial','condition'])['response'].mean()['T2']
    #for bar in ax[0,1].patches:
    #    if bar.get_height() == val[options[1]]:
    #        bar.set_color(c1)
    
    sns.despine(ax=ax[0,1], top = True, right = True)
    ax[0,1].get_xaxis().set_visible(False)
    
    legend_elements = [
    Patch(facecolor=c[0], label=l[0]),
    Patch(facecolor=c[1], label=l[1])
    ]
    
    ax[0,1].legend(handles=legend_elements, loc='upper left', fontsize=leg_s)
    ax[0,1].set_ylim((0, l_max))
    ax[0,1].set_yticks(np.arange(0,l_max, 20))

    ax[0,1].set_ylabel('')
    
    sns.boxplot(x='trial', y='RT', 
                data=df_filtered, 
                hue='condition', hue_order=options,
                medianprops={'color':'black'},
                fliersize=3,

                #showmeans=True, 
                #meanprops={"marker":">","markerfacecolor":"orange"},
                #errorbar='se',
                #err_kws={"color": "gray", 'linewidth': 1.2},
                #capsize = 0.1,
                palette=c,
                ax=ax[1,1])
    
    for i, box in enumerate(ax[1,1].artists):
        if i == 1:
            box.set_facecolor(c1)
            
    #val = df_filtered.groupby(['trial','condition'])['RT'].mean()['T2']
    #for bar in ax[1,1].patches:
    #    if np.isclose(bar.get_height(), val[options[1]], atol=0.01):
    #        bar.set_color(c1)
    
    sns.despine(ax=ax[1,1], top = True, right = True)
    ax[1,1].set_ylim((0, 8))
    ax[1,1].set_ylabel('')
    ax[1,1].set_xlabel('')
    
    ax[1,0].get_legend().remove()
    ax[1,1].get_legend().remove()
    
    plt.tight_layout()
    plt.savefig(f, dpi=600)

adapt_plots(df_response, 'Figures/Figure 3a.png')

x = df_stp.loc[(df_stp["prev"]=='Up'),   "few-shot"]
y = df_stp.loc[(df_stp["prev"]=='Fixed'),"few-shot"]
z = df_stp.loc[(df_stp["prev"]=='Down'), "few-shot"]

for name, data in zip(["Up", "Fixed", "Down"], [x, y, z]):
    print(f"{name}, {np.median(data)}")
    stat = ttest(data, 0, paired=False, alternative='greater')
    t = stat['T'][0]
    df = stat['dof'][0]
    p = stat['p-val'][0]
    BF10 = stat['BF10'][0]
    print(f"- one-tailed p = {p}, t({df}) = {t:.2f}, BF10 = {BF10}")

#stats...
x = df_response.loc[(df_response["condition"]==6) & (df_response["trial"]=="T3"),"RT"]
y = df_response.loc[(df_response["condition"]==4) & (df_response["trial"]=="T3"),"RT"]

t ,pval = ttest_ind(np.log10(x), np.log10(y),
                    equal_var=True,
                    alternative='less')
bf = bayesfactor_ttest(t, np.size(x), np.size(y), 
                       paired=False, 
                       alternative='less',
                       r=.5)

print(f"fixed 20: {np.mean(x):.3f} ± {np.std(x, ddof=1):.3f}, decreasing: {np.mean(y):.3f} ± {np.std(y, ddof=1):.3f}")
print(f'one-tailed t-test fixed < increasing: p = {pval:.5f}, t = {t:.2f}, BF = {bf:.3f}')

res = permutation_test((x, y), statistic, 
                       vectorized=False,
                       n_resamples = 9999, 
                       alternative='less')

print(f"fixed 20: {np.median(x)}, decreasing: {np.median(y)}")
print(f"medians diff {res.statistic}, p(prem.) = {res.pvalue:.4f}")

'''
analysis by prior groups
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
    #plt.show()
    plt.savefig(file, dpi=600)
    
df = pd.read_csv('Data/Experiment 9 - preregistered/merged - filtered.csv')
options = [2, 3, 7]
df_stp = df[df['Condition'].isin(options)]
df_stp['0.Line Length.0.rt'] = df_stp['0.Line Length.0.rt'] / 1000
df_stp['2.Line Length.0.rt'] = df_stp['2.Line Length.0.rt'] / 1000
df_stp['4.Line Length.0.rt'] = df_stp['4.Line Length.0.rt'] / 1000
df_stp['3.Recent Prior Distorted.0.rt'] = df_stp['3.Recent Prior Distorted.0.rt'] / 1000
df_stp['1.Recent Prior Normal.0.rt'] = df_stp['1.Recent Prior Normal.0.rt'] / 1000
df_stp['few-shot'] = (df_stp["4.Line Length.0.answer"] - df_stp["0.Line Length.0.answer"])*100 / df_stp["0.Line Length.0.answer"]

#df_stp.loc[df_stp["STP"]!="Weak"]
plot_prior(df_stp, f'Figures/Figure s1_all.png')


#preregistration: learning of ex>internal
x = df_stp.loc[(df_stp["STP"]=='External') ,"one-shot"]
y = df_stp.loc[(df_stp["STP"]=='Internal') ,"one-shot"]

stat = ttest(x, y, alternative='greater')
t = stat['T'][0]
p = stat['p-val'][0]
BF10 = stat['BF10'][0]
print(f"External: {np.mean(x):.1f}%, Internal: {np.mean(y):.1f}%\none-tailed t-test:\np = {np.round(p,decimals=5)}, t = {t:.2f}, BF10 = {BF10}")
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
----------------------draft-----------------------
def plot_stp(df, i, file): 
    fig = plt.figure()
    
    line_w = 2.5
    
    palette = {'External': 'lightsteelblue', 
               'Internal': 'mediumblue', 
               'Weak': 'darkblue'}

    sns.boxplot(
        x="STP", y="few-shot", data=df,
        fliersize=3,
        palette=palette, 
        order=['External', 'Internal', 'Weak'],)
        #edgecolor="black", ,

    plt.ylim(-100, 300)
    label = 'Change [%]\n(' + r'$\frac{T3 - T1}{T1}$' + ')'
    plt.ylabel(label)
    #n_weak, n_ext, n_int = df["STP"].value_counts()['Weak'], df["STP"].value_counts()['External'], df["STP"].value_counts()['Internal']
    
    #ax3.set_xlabel(f'{conditions_names[i]} (N = {n_ext}, {n_int}, {n_weak})')

    sns.despine(top = True, right = True)
    plt.axhline(y=0, c='k', linestyle='-')
    plt.tight_layout()
    #plt.show()
    plt.savefig(file, dpi=500)


def plot_phyen(df, i, file): 
    fig = plt.figure()
    gs = fig.add_gridspec(2,2)
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])
    ax3 = fig.add_subplot(gs[1,:])
    line_w = 2.5
    palette = {'Plain': 'lightcoral', 
               #'Internal': 'orangered', 
               'Design': 'firebrick'}
    #baseline
    df_long = df.melt(id_vars=['participant #', '1.Recent Prior Normal.0.answer'], 
                       value_vars=['0.Line Length.0.answer', '2.Line Length.0.answer', '4.Line Length.0.answer'], 
                       var_name='Measure', 
                       value_name='Value')
    
    sns.pointplot(data=df_long, x='Measure', y='Value', 
                  hue='1.Recent Prior Normal.0.answer', 
                  hue_order=['Plain', 'Design'],
                  errorbar='se',
                  linewidth=line_w,
                  err_kws={"color": "k", 'linewidth': 1.2},
                  capsize = 0.1,
                  palette=palette,
                  dodge=True, markers='o',
                  ax=ax1)
    
    #ax1.get_legend().remove()
    ax1.legend(title='Group', loc='upper left', fontsize='small')
    ax1.set_xlabel("")
    ax1.set_xticks([0, 1, 2], ['T1','T2','T3'])
    ax1.set_ylim(0, 60)
    ax1.set_ylabel('Legnth estimate\n[# of underbars]')
    sns.despine(ax=ax1, top = True, right = True)


    df_long_rt = df.melt(id_vars=['participant #', 'STP', "1.Recent Prior Normal.0.answer"], 
                       value_vars=['0.Line Length.0.rt', '2.Line Length.0.rt', '4.Line Length.0.rt'], 
                       var_name='Measure', 
                       value_name='Value')
    
    sns.pointplot(data=df_long_rt, x='Measure', y='Value', 
                  hue='1.Recent Prior Normal.0.answer', 
                  errorbar='se',
                  linewidth=line_w,
                  err_kws={"color": "k", 'linewidth': 1.2},
                  capsize = 0.1,
                  hue_order=['Plain', 'Design'],
                  palette=palette,
                  dodge=True, markers='o',
                  ax=ax2)
                 #errorbar='se', errwidth=1.2, capsize = 0.2,
    ax2.set_xticks([0, 1, 2], ['T1','T2','T3'])
    ax2.set_ylim(0, 6)
    ax2.set_xlabel("")
    ax2.set_ylabel('Response time\n[s]')
    sns.despine(ax=ax2, top = True, right = True)
    ax2.get_legend().remove()


    sns.boxplot(
        x="1.Recent Prior Normal.0.answer", y="few-shot", data=df,
        fliersize=3,
        palette=palette, 
        order=['Plain', 'Design'],
        ax=ax3)
        #edgecolor="black", ,

    ax3.set_ylim(-200, 400)
    label = 'Change [%]\n(' + r'$\frac{T3 - T1}{T1}$' + ')'
    ax3.set_ylabel(label)
    #n_weak, n_ext, n_int = df["STP"].value_counts()['Weak'], df["STP"].value_counts()['External'], df["STP"].value_counts()['Internal']
    
    #ax3.set_xlabel(f'{conditions_names[i]} (N = {n_ext}, {n_int}, {n_weak})')

    sns.despine(ax=ax3, top = True, right = True)

    plt.tight_layout()
    #plt.show()
    plt.savefig(file, dpi=500)

#agg plot
df = pd.read_csv('Data/Experiment 9 - preregistered/merged - filtered.csv')
options = [2, 3, 4, 5, 7] #without scales 15 and 20 (smaller learning effect)
df_stp = df[df['Condition'].isin(options)]
#change ms units to s 
df_stp['0.Line Length.0.rt'] = df_stp['0.Line Length.0.rt'] / 1000
df_stp['2.Line Length.0.rt'] = df_stp['2.Line Length.0.rt'] / 1000
df_stp['4.Line Length.0.rt'] = df_stp['4.Line Length.0.rt'] / 1000

#calculating the change aka learning effect
df_stp['few-shot'] = (df_stp["4.Line Length.0.answer"] - df_stp["0.Line Length.0.answer"])*100 / df_stp["0.Line Length.0.answer"]

df_plot = df_stp
plot_phyen(df_plot, 8, f'Figures/with MOPP/Figure s2.png')

df_plot = df_stp[df_stp['STP']!='Internal']
plot_stp(df_plot, 8, f'Figures/Figure s1.png')

#plot by condition i
for i in options:
    df_i = df_stp.loc[df_stp['Condition'] == i]
    plot_stp(df_i, i, f'Figures/with MOPP/Figure 3_{i}.png')

'''
