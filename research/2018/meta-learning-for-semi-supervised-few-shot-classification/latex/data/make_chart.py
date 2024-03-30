"""
A bar plot with errorbars and height labels on individual bars
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import jnet_bar


def plot_one(ax,
             data,
             labels,
             title,
             min,
             max,
             y_label=True,
             legend=True,
             x_label=True,
             add_text=False):
    # fig = plt.figure(figsize=(18, 6))
    ind = np.arange(len(labels))  # the x locations for the groups
    width = 1 / 5.0  # the width of the bars
    colors = ['r', 'brown', 'orange', 'c']
    rects1 = ax.bar(
        ind,
        data['Baseline'][0],
        width,
        color=colors[0],
        yerr=data['Baseline'][1],
        ecolor='k',
        error_kw={'elinewidth': 3,
                  'capthick': 2})
    rects2 = ax.bar(
        ind + width,
        data['K-Means'][0],
        width,
        color=colors[1],
        yerr=data['K-Means'][1],
        ecolor='k',
        error_kw={'elinewidth': 3,
                  'capthick': 2})
    rects3 = ax.bar(
        ind + 2 * width,
        data['Extra-Radius'][0],
        width,
        color=colors[2],
        yerr=data['Extra-Radius'][1],
        ecolor='k',
        error_kw={'elinewidth': 3,
                  'capthick': 2})
    rects4 = ax.bar(
        ind + 3 * width,
        data['Mean-Shift'][0],
        width,
        color=colors[3],
        yerr=data['Mean-Shift'][1],
        ecolor='k',
        error_kw={'elinewidth': 3,
                  'capthick': 2})

    # add some text for labels, title and axes ticks
    if legend:
        ax.legend(
            (rects1[0], rects2[0], rects3[0], rects4[0]),
            ('Supervised', 'Soft K-Means', 'Soft K-Means+Cluster',
             'Masked Soft K-Means'),
            # loc=2
            loc=0)
        plt.setp(ax.get_legend().get_texts(), fontsize=16)
    if y_label:
        ax.set_ylabel('Test Acc. (%)', fontsize=20)
    else:
        # ax.get_yaxis().set_visible(False)
        ax.set_yticklabels([])
    if x_label:
        ax.set_xlabel('Number of Unlabeled Items Per Class', fontsize=20)
        ax.set_xticks(ind + width * 2)
        ax.set_xticklabels(jnet_bar.num_unlabeled)
    else:
        # ax.get_xaxis().set_visible(False)
        ax.set_xticklabels([])

    ax.set_axisbelow(True)
    ax.grid(color='k', linestyle=':', linewidth=1)
    ax.set_title(title, fontsize=24)
    ax.set_ylim(min, max)

    zed = [
        tick.label.set_fontsize(16)
        for tick in ax.get_yaxis().get_major_ticks()
    ]
    zed = [
        tick.label.set_fontsize(16)
        for tick in ax.get_xaxis().get_major_ticks()
    ]

    if add_text:

        def autolabel(rects):
            """
            Attach a text label above each bar displaying its height
            """
            for rect in rects:
                height = rect.get_height()
                ax.text(
                    rect.get_x() + rect.get_width() / 2.,
                    height + 0.5,
                    '%.2f' % height,
                    ha='center',
                    va='bottom')

        autolabel(rects1)
        autolabel(rects2)
        autolabel(rects3)
        autolabel(rects4)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(20, 10))
    ax = axes.flatten()
    plot_one(
        ax[0],
        jnet_bar.nd1_results,
        jnet_bar.num_unlabeled,
        '1-shot w/o Distractors',
        43,
        55,
        x_label=False,
        legend=True)
    plot_one(
        ax[1],
        jnet_bar.d1_results,
        jnet_bar.num_unlabeled,
        '1-shot w/ Distractors',
        43,
        55,
        x_label=False,
        y_label=False,
        legend=False)
    plot_one(
        ax[2],
        jnet_bar.nd5_results,
        jnet_bar.num_unlabeled,
        '5-shot w/o Distractors',
        65,
        71,
        legend=False)
    plot_one(
        ax[3],
        jnet_bar.d5_results,
        jnet_bar.num_unlabeled,
        '5-shot w/ Distractors',
        65,
        71,
        y_label=False,
        legend=False)

    plt.tight_layout()
    plt.savefig('../figures/tnet_num_unlabel.pdf')

    fig, axes = plt.subplots(2, 2, figsize=(20, 10))
    ax = axes.flatten()
    plot_one(
        ax[0],
        jnet_bar.nd1_results,
        jnet_bar.num_unlabeled,
        '1-shot w/o Distractors',
        43,
        55,
        x_label=False,
        add_text=True,
        legend=True)
    plot_one(
        ax[1],
        jnet_bar.d1_results,
        jnet_bar.num_unlabeled,
        '1-shot w/ Distractors',
        43,
        55,
        x_label=False,
        y_label=False,
        add_text=True,
        legend=False)
    plot_one(
        ax[2],
        jnet_bar.nd5_results,
        jnet_bar.num_unlabeled,
        '5-shot w/o Distractors',
        65,
        71,
        legend=False,
        add_text=True)
    plot_one(
        ax[3],
        jnet_bar.d5_results,
        jnet_bar.num_unlabeled,
        '5-shot w/ Distractors',
        65,
        71,
        y_label=False,
        legend=False,
        add_text=True)

    plt.tight_layout()
    plt.savefig('../figures/tnet_num_unlabel_text.pdf')


if __name__ == '__main__':
    main()
