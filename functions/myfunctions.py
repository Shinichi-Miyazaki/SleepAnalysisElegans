#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 22:14:30 2020
@author: miyazakishinichi
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import os.path
import numpy as np
import itertools
import seaborn


# lethargus analyzer
def maxisland_start_len_mask(a, fillna_index=-1, fillna_len=0):
    # a is a boolean array

    pad = np.zeros(a.shape[1], dtype=bool)
    mask = np.vstack((pad, a, pad))

    mask_step = mask[1:] != mask[:-1]
    idx = np.flatnonzero(mask_step.T)
    island_starts = idx[::2]
    island_lens = idx[1::2] - idx[::2]
    n_islands_percol = mask_step.sum(0) // 2

    bins = np.repeat(np.arange(a.shape[1]), n_islands_percol)
    scale = island_lens.max() + 1

    scaled_idx = np.argsort(scale * bins + island_lens)
    grp_shift_idx = np.r_[0, n_islands_percol.cumsum()]
    max_island_starts = island_starts[scaled_idx[grp_shift_idx[1:] - 1]]

    max_island_percol_start = max_island_starts % (a.shape[0] + 1)

    valid = n_islands_percol != 0
    cut_idx = grp_shift_idx[:-1][valid]
    max_island_percol_len = np.maximum.reduceat(island_lens, cut_idx)

    out_len = np.full(a.shape[1], fillna_len, dtype=int)
    out_len[valid] = max_island_percol_len
    out_index = np.where(valid, max_island_percol_start, fillna_index)
    return out_index, out_len, island_starts, island_lens


def lethargus_analyzer(data, body_size):
    # out of lethargus = 5 minutes after lethargus end
    out_duration = 300
    # make resuld folder
    os.makedirs("./results", exist_ok=True)
    os.chdir("./results")

    # Arange column name
    data.columns = ["worm" + str(i+1) for i in range(data.shape[1])]

    # make time axis
    data["time_axis(min)"] = [a for a in range(len(data))]
    data = data.set_index(['time_axis(min)'])

    # make boolean array
    # if the activity > 1% of the body, Wake
    Wake_sleep_boolean = data<body_size/100
    Wake_sleep_boolean.to_csv('./Wake_sleep_boolean.csv')

    # calculate FoQ
    FoQ_data = Wake_sleep_boolean.rolling(300, min_periods=1, center=True).mean()
    FoQ_data.to_csv('./FoQ_data.csv')

    # Make FoQ figures
    seaborn.set_style(style="ticks")
    os.makedirs("./figures", exist_ok=True)
    for k in FoQ_data.columns:
        plt.figure()
        FoQ_data[k].plot.line(style="k")
        plt.savefig('./figures/{}.png'.format(k))
        plt.show()

    # detect lethargus
    # for searching letahrgus enter and end, make boolean array
    # if the FoQ > 0.05 True, if not False
    lethargus_judge_boolean = FoQ_data>0.05
    lethargus_judge_boolean.to_csv('./Lethargus_judge_boolean.csv')

    max_start, max_length, all_start, all_length = maxisland_start_len_mask(lethargus_judge_boolean)

    # quiescent bout analysis
    Sleep_bout_results = maxisland_start_len_mask(Wake_sleep_boolean)
    Sleep_bout_durations = Sleep_bout_results[3]
    Sleep_bout_starts = Sleep_bout_results[2]

    # wake bout analysis
    Wake_bout_results = maxisland_start_len_mask(~Wake_sleep_boolean)
    Wake_bout_durations = Wake_bout_results[3]
    Wake_bout_starts = Wake_bout_results[2]

    def each_column_analysis():
        # Q_durations is all the quiescent bouts
        column_name = ['bodysize', 'FoQ_during_Lethargus', 'FoQ_out',
                       'duration (hr)', 'interpletation 0 is adequate',
                       'Mean Quiescent Bout (sec)', 'Mean Active Bout (sec)',
                       'Transitions (/hr)', "Total Q (sec)", "Total A (sec)"]
        result = []
        column_num = data.shape[0]
        row_num = data.shape[1]

        lethargus_dataframe = pd.DataFrame()

        for i in range(row_num):
            num = i + 1
            # extract quiescent island indices
            temp_area_indices = list(itertools.chain.from_iterable \
                                         (np.where((all_start >= column_num * i) & \
                                                   (all_start <= column_num * num))))
            # quiescent island length
            quiescent_lengths = all_length[temp_area_indices]
            # count only the islands which is longer than 1hour
            quiescent_island_num = np.count_nonzero(quiescent_lengths > 1800)

            # max island (= lethargus) end
            max_q_end = max_start[i] + max_length[i]
            # out of lethargus (5 min after lethargus end)
            max_q_out = max_q_end + out_duration
            # tempFoQ is FoQ series of current chamber
            temp_foq = FoQ_data.iloc[:, i]
            # calculate average FOQ during lethargus
            foq_mean = sum(temp_foq[max_start[i]:max_q_end])/max_length[i]
            # calculate average FoQ out of lethargus
            foq_out = sum(temp_foq[max_q_end:max_q_out])/ out_duration
            # calculate lethargus length
            lethargus_length = max_length[i] / 1800

            # init value
            judge, mean_q_duration, mean_a_duration, transitions, \
                total_q, total_a = 0, 0, 0, 0, 0, 0

            # check start point & end point
            if quiescent_island_num > 1:
                judge = "multiple quiescent islands"
            elif quiescent_island_num == 0:
                judge = "there is no lethargus period"
            elif max_start[i] < 1800:
                judge = "Lethargus start within first 1 hour"
            elif max_q_end > column_num - 900:
                judge = "Lethargus did not end"
            else:
                judge = "applicable for lethargus analysis"
                # extract from 1hour before lethargus to the end
                LeFoQdf = temp_foq[max_start[i] - 1800:]
                lethargus_dataframe = pd.concat([lethargus_dataframe, LeFoQdf], axis=1)

                q_starts_index = np.where((Sleep_bout_starts - column_num * i > max_start[i]) \
                                          & (Sleep_bout_starts - column_num * i < max_q_end))
                a_starts_index = np.where((Wake_bout_starts - column_num * i > max_start[i]) \
                                          & (Wake_bout_starts - column_num * i < max_q_end))
                q_starts_lethargus = Sleep_bout_starts[q_starts_index] - column_num * i
                a_starts_lethargus = Wake_bout_starts[a_starts_index] - column_num * i

                # calculate total Q
                q_durations_lethargus = Sleep_bout_durations[q_starts_index]
                total_q = np.sum(q_durations_lethargus) * 2
                # calculate total A
                a_durations_lethargus = Wake_bout_durations[a_starts_index]
                total_a = np.sum(a_durations_lethargus) * 2
                # calculate mean Q
                mean_q_duration = np.mean(q_durations_lethargus) * 2
                # calculate mean A
                mean_a_duration = np.mean(a_durations_lethargus) * 2
                # averaged transitions
                transitions = len(q_durations_lethargus) / lethargus_length

                # calucurate parameters each 15 min bins
                # 3 hour -> 12 bins
                os.makedirs("./subdivided/worm{}".format(num), exist_ok=True)
                bins = 6
                if max_length[i] < 6000:
                    bins = max_length[i] // 900 - 1
                else:
                    pass
                ex_start = max_start[i]
                if a_starts_lethargus[0] > q_starts_lethargus[0]:
                    q_durations_lethargus = q_durations_lethargus[1:]
                # if A is more than Q, A is deleted
                if len(a_durations_lethargus) > len(q_durations_lethargus):
                    a_durations_lethargus = a_durations_lethargus[:-1]
                Lethargus_QandA = np.stack((a_durations_lethargus,
                                            q_durations_lethargus), 1)

                for j in range(bins):
                    start = j * 900 + ex_start
                    end = (j + 1) * 900 + ex_start
                    ex_Leth_QandA = Lethargus_QandA[np.where((a_starts_lethargus > start) & \
                                                             (a_starts_lethargus < end))]
                    ex_Leth_QandA = pd.DataFrame(ex_Leth_QandA, columns=["A", "Q"])
                    ex_Leth_QandA.to_csv("./subdivided/worm{0}/leth_{1}.csv".format(num, j))

            temp_result = np.array([body_size, foq_mean, foq_out,
                                    lethargus_length, judge, mean_q_duration,
                                    mean_a_duration, transitions, total_q,
                                    total_a])
            result.append(temp_result)
        result_df = pd.DataFrame(result, index=["worm" + str(i+1) for i in range(row_num)],
                                 columns=column_name)
        result_df.to_csv('./result_summary.csv')
        lethargus_dataframe.to_csv('./lethargus_dataframe.csv')

    each_column_analysis()

