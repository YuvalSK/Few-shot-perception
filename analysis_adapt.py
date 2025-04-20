# -*- coding: utf-8 -*-
"""
perceptual learning of visual length
"""

import pandas as pd
import numpy as np
from scipy.stats import median_abs_deviation, sem, ks_2samp, wilcoxon, bartlett, fligner, levene, f_oneway, norm, ttest_ind, ttest_1samp, ttest_ind_from_stats, permutation_test, spearmanr, mannwhitneyu, shapiro
from pingouin import bayesfactor_ttest, welch_anova, ttest

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
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

options = [1, 2, 3, 6, 7]
df_fixed = df[df['Condition'].isin(options)]

#filter participants with missing trials 
df_fixed = df_fixed[df_fixed['2.Line Length.0.answer'].notnull()]

#one-shot stats
df_fixed['one-shot'] = (df_fixed["2.Line Length.0.answer"] - df_fixed["0.Line Length.0.answer"])*100 / df_fixed["0.Line Length.0.answer"]
df_fixed['one-shot'].groupby(df_fixed["scale"]).median()

x1 = df_fixed.loc[(df_fixed["scale"]==15),   "one-shot"]
x2 = df_fixed.loc[(df_fixed["scale"]==20),   "one-shot"]
x3 = df_fixed.loc[(df_fixed["scale"]==30),   "one-shot"]
x4 = df_fixed.loc[(df_fixed["scale"]==40),   "one-shot"]
x5 = df_fixed.loc[(df_fixed["scale"]==60),   "one-shot"]

for name, data in zip(["15", "20", "30", "40", "60"], [x1, x2, x3, x4, x5]):
    print(f"One-shot:\nFixed {name} (N = {len(data)}): x͂ = {np.median(data):.1f}% ± {median_abs_deviation(data):.1f}")
    stat = ttest(data, 0, paired=False, alternative='greater', correction=True)
    t = stat['T'][0]
    d = stat['dof'][0]
    p = stat['p-val'][0]
    BF10 = stat['BF10'][0]
    print(f"- one-tailed p = {np.round(p,decimals=5)}, t({d}) = {np.round(t,decimals=2)}, BF10 = {BF10}")

x = df_fixed.loc[(df_fixed["scale"]==15), "T1 ratio"]
y = df_fixed.loc[(df_fixed["scale"]==60),"T1 ratio"]
print(f"{np.median(x):.2f} ± {median_abs_deviation(x):.2f}")
print(f"{np.median(y):.2f} ± {median_abs_deviation(y):.2f}")

stat = ttest(x, y, paired=False, alternative='two-sided')
t = stat['T'][0]
d = stat['dof'][0]
p = stat['p-val'][0]
BF10 = stat['BF10'][0]
print(f"two-tailed t-test: p = {np.round(p,decimals=5)}, t({d:.0f}) = {t:.2f}, BF10 = {BF10}")


#one-shot effect is there across conditions/scales

df_fixed['two-shot'] = (df_fixed["4.Line Length.0.answer"] - df_fixed["0.Line Length.0.answer"])*100 / df_fixed["0.Line Length.0.answer"]
df_fixed["two-shot"].groupby(df_fixed["scale"]).median()

x1 = df_fixed.loc[(df_fixed["scale"]==15),   "two-shot"]
x2 = df_fixed.loc[(df_fixed["scale"]==20),   "two-shot"]
x3 = df_fixed.loc[(df_fixed["scale"]==30),   "two-shot"]
x4 = df_fixed.loc[(df_fixed["scale"]==40),   "two-shot"]
x5 = df_fixed.loc[(df_fixed["scale"]==60),   "two-shot"]
for name, data in zip(["15", "20", "30", "40", "60"], [x1, x2, x3, x4, x5]):
    print(f"Two-shot\nFixed {name} (N = {len(data)}): x͂ = {np.median(data):.1f}, x̄ = {np.mean(data):.1f} ± {sem(data, ddof=1):.1f}")
    stat = ttest(data, 0, paired=False, alternative='greater')
    t = stat['T'][0]
    d = stat['dof'][0]
    p = stat['p-val'][0]
    BF10 = stat['BF10'][0]
    print(f"- one-tailed p = {np.round(p,decimals=4)}, t({d}) = {np.round(t,decimals=2)}, BF10 = {BF10}")

#two-shot effect is there across conditions/scales

options = [6, 7, 4, 5]
df_fixed_adapt = df[df['Condition'].isin(options)].copy()
df_fixed_adapt['two-shot'] = (df_fixed_adapt["4.Line Length.0.answer"] - df_fixed_adapt["0.Line Length.0.answer"])*100 / df_fixed_adapt["0.Line Length.0.answer"]
df_fixed_adapt["two-shot"].groupby(df_fixed_adapt["scale"]).median()

def modify_prev(row):
    if row['prev'] == 'Fixed':
        return f'Fixed {row["scale"]}'
    return row['prev']

df_fixed_adapt.loc[:, 'category'] = df_fixed_adapt.apply(modify_prev, axis=1)

fig = plt.figure(figsize=(10, 6))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2], wspace=0.2)
ax1 = fig.add_subplot(gs[0, 0])  # Top-left: High adapt & Fixed 20
ax2 = fig.add_subplot(gs[0, 1])  # Top-right: Fixed 40 & Low adapt
ax3 = fig.add_subplot(gs[1, :])  # Bottom: Fixed 15–60

palette_top1 = {'Up': 'green', 'Fixed 20': 'gray'}
sns.boxplot(
    x="category", y="two-shot", hue="category",
    data=df_fixed_adapt[df_fixed_adapt["category"].isin(['Up', 'Fixed 20'])],
    palette=palette_top1,
    medianprops={'color': 'black'},
    fliersize=3,
    legend=False,
    ax=ax1
)

ax1.set_xticks(ax1.get_xticks())
ax1.set_xticklabels(['High adapt\n[20, 40, 20]', 'Fixed 20'])
ax1.set_ylabel('Change [%]\n(' + r'$\frac{T3 - T1}{T1}$' + ')')
ax1.set_ylim(-100, 600)
ax1.set_xlabel('')
ax1.set_yticks(np.arange(-100, 601, 100))
ax1.axhline(0, color='k')
sns.despine(ax=ax1)

palette_top2 = {'Fixed 40': 'dimgray', 'Down': 'red'}
sns.boxplot(
    x="category", y="two-shot", hue="category",
    data=df_fixed_adapt[df_fixed_adapt["category"].isin(['Fixed 40', 'Down'])],
    palette=palette_top2,
    medianprops={'color': 'black'},
    fliersize=3,
    legend=False,
    ax=ax2
)

ax2.set_xticks(ax2.get_xticks())
ax2.set_xticklabels(['Fixed 40', 'Low adapt\n[40, 20, 40]'])
ax2.set_ylabel('')
ax2.set_xlabel('')
ax2.set_ylim(-100, 600)
ax2.set_yticks(np.arange(-100, 601, 100))
ax2.axhline(0, color='k')
sns.despine(ax=ax2)

palette_bottom = {
    15: 'lightgray', 20: 'gray', 30: 'dimgray',
    40: 'dimgray', 60: 'dimgray'
}

sns.boxplot(
    x="scale", y="two-shot", hue="scale",
    data=df_fixed,
    palette=palette_bottom,
    order=[15, 20, 30, 40, 60],
    medianprops={'color': 'black'},
    fliersize=3,
    legend=False,
    ax=ax3
)

#cosmetics...
ax3.set_xticks(ax3.get_xticks())
ax3.set_xticklabels(['Fixed 15', 'Fixed 20', 'Fixed 30', 'Fixed 40', 'Fixed 60'])
ax3.set_ylabel('Change [%]\n(' + r'$\frac{T3 - T1}{T1}$' + ')')
ax3.set_xlabel('Conditions')
ax3.set_ylim(-100, 600)
ax3.set_yticks(np.arange(-100, 601, 100))
ax3.axhline(0, color='k')
sns.despine(ax=ax3)

ax1.text(-0.15, 1.05, '(a)', 
        transform=ax1.transAxes,
        fontsize=12, 
        fontweight='bold', 
        va='top', 
        ha='right')
ax2.text(-0.125, 1.05, '(b)', 
        transform=ax2.transAxes,
        fontsize=12, 
        fontweight='bold', 
        va='top', 
        ha='right')
ax3.text(-0.065, 1.05, '(c)', 
        transform=ax3.transAxes,
        fontsize=12, 
        fontweight='bold', 
        va='top', 
        ha='right')
    
plt.savefig('Figures/Figure 3_py.png', dpi=600)

x = df_fixed_adapt.loc[(df_fixed_adapt["prev"]=='Up'),   "two-shot"]
y = df_fixed_adapt.loc[(df_fixed_adapt["prev"]=='Fixed'),"two-shot"]
z = df_fixed_adapt.loc[(df_fixed_adapt["prev"]=='Down'), "two-shot"]

stat, p = f_oneway(x,y,z, nan_policy='omit')
print(f"ANOVA: F = {stat}, p = {p}")

#stat_l, pval_l = levene(x, y, z, center='median', nan_policy='omit')
#print(f"Levene's test: {pval_l}")
#stat_b, pval_b = bartlett(x, y, z)
#print(f"Brown-Forsythe test: {stat_b} p-value:{pval_b}")
#w = welch_anova(df_fixed, dv="two-shot", between = "prev")
#print(f"Welch correction ANOVA: {w['F'][0]:.2f} p-value:{w['p-unc'][0]}")

x = df_fixed_adapt.loc[(df_fixed_adapt["Condition"]==5), "two-shot"]
y = df_fixed_adapt.loc[(df_fixed_adapt["Condition"]==7),"two-shot"]

stat = ttest(x, y, paired=False, alternative='two-sided')
t = stat['T'][0]
d = stat['dof'][0]
p = stat['p-val'][0]
BF10 = stat['BF10'][0]
print(f"{conditions_names[5]} vs. {conditions_names[7]}\n- two-tailed t-test: p = {np.round(p,decimals=5)}, t({d:.0f}) = {t:.2f}, BF10 = {BF10}")

x = df_fixed_adapt.loc[(df_fixed_adapt["Condition"]==4), "two-shot"]
y = df_fixed_adapt.loc[(df_fixed_adapt["Condition"]==6),"two-shot"]

stat = ttest(x, y, paired=False, alternative='two-sided')
t = stat['T'][0]
d = stat['dof'][0]
p = stat['p-val'][0]
BF10 = stat['BF10'][0]
print(f"{conditions_names[4]} vs. {conditions_names[6]}\n- two-tailed t-test: p = {np.round(p,decimals=5)}, t({d:.0f}) = {t:.2f}, BF10 = {BF10}")

for name, data in zip(["Adapt", "Fixed"], [x, y]):
    print(f"{name}: x͂ = {np.median(data):.1f}, x̄ = {np.mean(data):.1f} ± {sem(data, ddof=1):.1f}")
    stat = ttest(data, 0, paired=False, alternative='greater')
    t = stat['T'][0]
    d = stat['dof'][0]
    p = stat['p-val'][0]
    BF10 = stat['BF10'][0]
    print(f"- one-tailed t-test: p = {np.round(p,decimals=5)}, t({df}) = {t:.2f}, BF10 = {BF10}")
    

#for mixed model analysis in JASP/R:
df_long = df.melt(id_vars=['participant #', 'STP', 'prev', 'scale'], 
                   value_vars=['0.Line Length.0.answer', '2.Line Length.0.answer', '4.Line Length.0.answer'], 
                   var_name='Trial', 
                   value_name='Estimate')
df_long.to_csv('Data/merged_long_raw.csv')


#################### draft code #####################
'''
x = df_response.loc[(df_response["condition"]==6) & (df_response["trial"]=="T3"),"few-shot"]
y = df_response.loc[(df_response["condition"]==4) & (df_response["trial"]=="T3"),"few-shot"]

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
