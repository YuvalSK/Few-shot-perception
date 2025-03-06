 -*- coding: utf-8 -*-
"""
the key of the fixed conditions:
options = [1, 6, 2, 7, 3] 
- f15, f20, f30, f40 and f60, respectively.
- note that in condition 6 (f20) the recent prior data is missing.  

the five prereg conditions:
options = [1, 2, 3, 4, 5] 
- f15, f30, f60, adapt up [20-40-20], adapt down [40-20-40]

- the pilot experiment is condition 8 with adapt up 15-10-15. 
It is without RT data, and the trials of the recent prior task (P1, P2) were presented on different pages 
(prior to MOPP)
"""
import pandas as pd
import numpy as np
from statsmodels.stats.anova import AnovaRM
from scipy.stats import f_oneway, norm, ttest_ind, ttest_1samp, ttest_ind_from_stats, permutation_test, spearmanr, mannwhitneyu
from pingouin import bayesfactor_ttest

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

def statistic(x, y):
    return np.median(x) - np.median(y)

def spearmanr_statistic(x, y):
    return spearmanr(x, y)[0]

conditions_names = {0: 'Pooled',
                    1:'F-15',
                    2: 'F-30',
                    3: 'F-60',
                    4: 'A-up [20-40-20]',
                    5: 'A-down [40-20-40]',
                    6: 'F-20',
                    7: 'F-40',
                    8: 'A-down [15-10-15]'}

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

def plot_fixed(df_response, options, colors):
    
    f, ax = plt.subplots(2, 1)
    sns.barplot(
        x="condition", 
        y="response", 
        hue="trial", 
        data=df_filtered, 
        errorbar="se", 
        edgecolor="black",
        errcolor="black",
        errwidth=1.0,
        capsize = 0.1,
        alpha=0.5,
        palette=colors, 
        order=options,
        ax=ax[0]
    )
    sns.stripplot(
        x="condition", 
        y="response", 
        hue="trial",
        marker="$\circ$", 
        ec="face", 
        s=8,
        order=options,
        palette=colors,
        legend=False,
        data=df_filtered, dodge=True, alpha=0.6, 
        ax=ax[0]
    )
    sns.barplot(
        x="condition", 
        y="RT", 
        hue="trial", 
        data=df_filtered, 
        errorbar="se", 
        edgecolor="black",
        errcolor="black",
        errwidth=1.0,
        capsize = 0.1,
        alpha=0.5,
        palette=colors, 
        order=options,
        ax=ax[1]
    )
    sns.stripplot(
        x="condition", 
        y="RT", 
        hue="trial",
        marker="$\circ$", 
        ec="face", 
        s=8,
        order=options,
        palette=colors,
        legend=False,
        data=df_filtered, dodge=True, alpha=0.6, 
        ax=ax[1]
    )
    ax[0].legend(title='Trial')
    ax[0].set(ylim=(0, 130))
    ax[0].set(yticks=np.arange(0, 130, 20))
    ax[0].get_xaxis().set_visible(False)
    ax[0].set_ylabel('Length estimate\n[# of unberbars]')
    sns.despine(ax=ax[0], top = True, right = True)

    ax[1].get_legend().remove()
    ax[1].set(ylim=(0, 9))
    ax[1].set(yticks=np.arange(0, 9,2))
    ax[1].set_xticklabels(['Fixed 15','Fixed 20','Fixed 30','Fixed 40','Fixed 60'])
    ax[1].set_ylabel('Response time\n[s]')
    ax[1].set_xlabel('Conditions')

    sns.despine(ax=ax[1], top = True, right = True)
    plt.tight_layout()
    plt.savefig('Figures/Figure 2_py.png', dpi=800)

#presenting all fixed conditions
options = [1, 6, 2, 7, 3]  
df_filtered = df_response[df_response['condition'].isin(options)]
colors = 'Greys'
df_filtered['RT'] = df_filtered['RT'] / 1000

plot_fixed(df_response, options, colors)

#analysis by preregistration of fixed conditions
options = [1, 2, 3]  
df_anova = df_response[df_response['condition'].isin(options)]
df_anova = df_anova.loc[(df_anova['trial'] =='T1') | (df_anova['trial'] =='T3')]

res = AnovaRM(
        data=df_anova,
        depvar = 'response',
        subject = 'id',
        within=['trial'],
        aggregate_func='mean'
        ).fit()

print(f"rmANOVA for conditions: {options}")
print(res)
print(f'\np-value: {res.anova_table["Pr > F"][0]} \nNote: without between subject effect of condition')