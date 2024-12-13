import numpy as np
import pandas as pd
import pingouin as pg
from pingouin import ttest, print_table

import seaborn
import matplotlib.pyplot as plt

df = pd.read_table("human_results.csv", delimiter=",")
# print(df)

print("***********")
ddpglfd_df = df[df['method'] == 'ddpglfd']
isolated_active_df = df[df['method'] == 'isolated-active']
episodic_active_df = df[df['method'] == 'active-sac']
metrics_list = ['Mental Demand', 'Physical Demand', 'Temporal Demand', 'Performance', 'Effort', 'Frustration', 'Human Time', 'Convergence Steps']

# parameters for plotting boxplots
labels = ['DDPG-LfD', 'I-ARLD', 'EARLY']

for metrics in metrics_list:
    res_ddpg_episodic = ttest(x=ddpglfd_df[metrics], y=episodic_active_df[metrics], paired=True)
    res_isolated_episodic = ttest(x=isolated_active_df[metrics], y=episodic_active_df[metrics], paired=True)
    print("[{}]: average score for ddpglfd: {}, std: {}".format(metrics, ddpglfd_df[metrics].mean(), ddpglfd_df[metrics].std()))
    print("[{}]: average score for isolated-active: {}, std: {}".format(metrics, isolated_active_df[metrics].mean(), isolated_active_df[metrics].std()))
    print("[{}]: average score for episodic-active: {}, std: {}".format(metrics, episodic_active_df[metrics].mean(), episodic_active_df[metrics].std()))
    print_table(res_ddpg_episodic)
    print_table(res_isolated_episodic)

    print("Results of Tukey post-hoc test:")
    tukey_res = pg.pairwise_tukey(data=df, dv=metrics, between='method').round(3)
    print_table(tukey_res)

    res_anova = pg.rm_anova(dv=metrics, within='method', subject='subject id', data=df, detailed=False)
    print_table(res_anova)

    # plot the boxplot
    # plt.clf()
    seaborn.set(style="whitegrid", font_scale=1.4)
    plt.figure(figsize=(7, 4))
    ax = seaborn.boxplot(x ='method', y =metrics, data = df, hue='method', fill=False, width=0.4, linewidth=1.7)
    ax.set_xticklabels(labels)
    ax.set_xlabel(None)

    if not (metrics == 'Human Time' or metrics == 'Convergence Steps'):
        ax.set(ylim=(-1, 24))
        ax.yaxis.set_ticks(np.arange(0, 24, 4))
    elif metrics == 'Human Time':
        ax.set_ylabel('Human Time (min)')
    else:
        ax.set_ylabel('Convergence Steps (×1000)')

    # add symbol for significance
    if metrics == 'Mental Demand':
        x1, x2 = 0, 2  # columns
        y, h, col = 21, 0.5, 'k'

        plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=0.8, c=col)
        plt.text((x1 + x2) * .5, y + h, "**", ha='center', va='bottom', color=col)

        x1, x2 = 1, 2  # columns
        y, h, col = 17, 0.5, 'k'

        plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=0.8, c=col)
        plt.text((x1 + x2) * .5, y + h, "*", ha='center', va='bottom', color=col)
    elif metrics == 'Convergence Steps':
        x1, x2 = 0, 2  # columns
        y, h, col = 125, 2, 'k'

        plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=0.8, c=col)
        plt.text((x1 + x2) * .5, y + h, "***", ha='center', va='bottom', color=col)

        x1, x2 = 1, 2  # columns
        y, h, col = 115, 2, 'k'

        plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=0.8, c=col)
        plt.text((x1 + x2) * .5, y + h, "*", ha='center', va='bottom', color=col)

        x1, x2 = 0, 1  # columns
        y, h, col = 105, 2, 'k'

        plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=0.8, c=col)
        plt.text((x1 + x2) * .5, y + h, "***", ha='center', va='bottom', color=col)
    elif metrics == 'Human Time':
        x1, x2 = 0, 2  # columns
        y, h, col = 19, 0.5, 'k'

        plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=0.8, c=col)
        plt.text((x1 + x2) * .5, y + h, "***", ha='center', va='bottom', color=col)

        x1, x2 = 1, 2  # columns
        y, h, col = 17, 0.5, 'k'

        plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=0.8, c=col)
        plt.text((x1 + x2) * .5, y + h, "***", ha='center', va='bottom', color=col)

        x1, x2 = 0, 1  # columns
        y, h, col = 15, 0.5, 'k'

        plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=0.8, c=col)
        plt.text((x1 + x2) * .5, y + h, "***", ha='center', va='bottom', color=col)



    plt.show()
    ax.get_figure().savefig('plots/' + metrics + '.png', dpi=300)