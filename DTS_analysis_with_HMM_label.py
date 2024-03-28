"""
Lethargus analysis
20230308
Author: Shinichi Miyazaki

This script enable lethargus analysis for imagesubtraction data
"""

import os
import tkinter.filedialog
import numpy as np
import pandas as pd
from functions.myfunctions import maxisland_start_len_mask
import itertools

# parameters
# imaging_interval (sec)
imaging_interval = 2 # (sec)
image_num_per_hour = int(3600 / imaging_interval)
# if DTS starts within 30 min, it is not used for analysis
margin_image_num = int(1800 / imaging_interval)
# The definition of out of DTS is 10 min after DTS end
out_DTS_duration = 600 # (sec)
out_DTS_image_num = int(out_DTS_duration / imaging_interval)
# Time window of rolling average
rolling_window_duration = 600 # (sec)
rolling_window_image_num = int(rolling_window_duration / imaging_interval)
# threshold of FoQ
FoQ_threshold = 0.05

def DTS_analysis_with_HMM_label(analysis_res_df,
                                FoQ_raw,
                                DTS_boolean,
                                HMM_label,
                                body_size):
    # lethargus bout analysis
    max_start, max_length, all_start, all_length = maxisland_start_len_mask(DTS_boolean)
    # quiescent bout analysis
    _, _, Sleep_bout_starts, Sleep_bout_durations = maxisland_start_len_mask(HMM_label.values)

    # wake bout analysis
    _, _, Wake_bout_starts, Wake_bout_durations = maxisland_start_len_mask(-(HMM_label.values-1))

    column_name = ['bodysize_pixel', 'FoQ_during_DTS_arb', 'FoQ_out_arb',
                   'duration_hour', 'interpretation',
                   'MeanSleepBout_sec', 'MeanAwakeBout_sec',
                   'Transitions_/hour', "TotalQ_sec", "TotalA_sec"]
    result = []
    column_num, row_num = analysis_res_df.shape
    DTS_df = pd.DataFrame()
    for i in range(row_num):
        num = i + 1
        # extract quiescent island indices
        temp_area_indices = list(itertools.chain.from_iterable \
                                     (np.where((all_start >= column_num * i) & \
                                               (all_start <= column_num * num))))
        # quiescent island length
        quiescent_lengths = all_length[temp_area_indices]
        # count only the islands which is longer than 1hour
        quiescent_island_num = np.count_nonzero(quiescent_lengths > image_num_per_hour)

        # max island (= lethargus) end
        max_q_end = max_start[i] + max_length[i]
        # out of lethargus (5 min after lethargus end)
        max_q_out = max_q_end + out_DTS_image_num
        # tempFoQ is FoQ series of current chamber
        temp_foq = FoQ_raw.iloc[:, i]
        # calculate average FOQ during lethargus
        foq_mean = temp_foq.iloc[max_start[i]:max_q_end].mean()
        # calculate average FoQ out of lethargus
        foq_out = temp_foq.iloc[max_q_end:max_q_out].mean()
        # calculate lethargus length
        lethargus_length = max_length[i] / image_num_per_hour

        # init value
        judge, mean_q_duration, mean_a_duration, transitions, \
            total_q, total_a = 0, 0, 0, 0, 0, 0

        # check start point & end point
        if quiescent_island_num > 1:
            judge = "Multiple_DTS"
        elif quiescent_island_num == 0:
            judge = "No_DTS"
        elif max_start[i] < margin_image_num:
            judge = "DTS_Start_Within_30min"
        elif max_q_end > column_num - margin_image_num:
            judge = "DTS_not_end"
        else:
            judge = "Applicable"
            # extract from  30 min before lethargus to the end
            LeFoQdf = temp_foq.iloc[max_start[i] - margin_image_num:]
            DTS_df = pd.concat([DTS_df, LeFoQdf.reset_index().iloc[:, 1]], axis=1)

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
            a_durations_lethargus = Wake_bout_durations[a_starts_index][:-1]
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
            if len(a_durations_lethargus) < len(q_durations_lethargus):
                q_durations_lethargus = q_durations_lethargus[:-1]
            Lethargus_QandA = np.stack((a_durations_lethargus,
                                        q_durations_lethargus), 1)

            for j in range(bins):
                start = j * 900 + ex_start
                end = (j + 1) * 900 + ex_start
                ex_Leth_QandA = Lethargus_QandA[np.where((a_starts_lethargus > start) & \
                                                         (a_starts_lethargus < end))]
                ex_Leth_QandA = pd.DataFrame(ex_Leth_QandA, columns=["A", "Q"])
                ex_Leth_QandA.to_csv("./subdivided/worm{0}/leth_{1}.csv".format(num, j))

            # make DTS boolean dataframe
            # DTS booleanのmax_startからmax_q_endまでの間をTrueにする 他はFALSEにする
            DTS_boolean["DTS_mask"] = 0
            DTS_boolean["DTS_mask"].iloc[max_start[i]:max_q_end] = 1
            DTS_boolean["Before_DTS_mask"] = 0
            DTS_boolean["Before_DTS_mask"].iloc[max_start[i]-300:max_start[i]] = 1
            DTS_boolean["After_DTS_mask"] = 0
            DTS_boolean["After_DTS_mask"].iloc[max_q_end:max_q_end + 300] = 1
            DTS_boolean.to_csv("./subdivided/worm{0}/DTS_mask.csv".format(num))



        temp_result = np.array([body_size, foq_mean, foq_out,
                                lethargus_length, judge, mean_q_duration,
                                mean_a_duration, transitions, total_q,
                                total_a])
        result.append(temp_result)
    result_df = pd.DataFrame(result, index=["worm" + str(i+1) for i in range(row_num)],
                             columns=column_name)
    result_df.to_csv('./result_summary.csv')
    DTS_df.to_csv('./Lethargus_dataframe.csv')


def main():
    # select file
    root = tkinter.Tk()
    root.withdraw()
    body_size = 100 # int(input('bodysize (pixel) を入力 : '))
    print("select area and DTS mask csv file")
    filepath = tkinter.filedialog.askopenfilename()
    os.chdir(os.path.dirname(filepath))

    analysis_res_df = pd.read_csv(filepath)["Area"]
    DTS_mask = pd.read_csv(filepath)["DTS_mask"]

    # make result folder
    os.makedirs("./results", exist_ok=True)
    os.chdir("./results")
    os.makedirs("./figures", exist_ok=True)

    # make time axis
    timeaxis = pd.Series([sec / (60 / imaging_interval) for sec in range(len(analysis_res_df))])
    analysis_res_df = pd.concat([timeaxis, analysis_res_df], axis=1)
    analysis_res_df.columns = ["time_axis(min)", "worm1"]
    analysis_res_df = analysis_res_df.set_index("time_axis(min)")

    # make boolean array
    # if the activity > 1% of the body, Wake
    Wake_sleep_boolean = analysis_res_df < body_size / 100
    Wake_sleep_boolean.to_csv('./Wake_sleep_boolean.csv')

    # calculate FoQ
    FoQ_raw = Wake_sleep_boolean.rolling(int(rolling_window_image_num), min_periods=1, center=True).mean()
    FoQ_raw.to_csv('./FoQ_data.csv')

    # detect lethargus
    # for searching letahrgus enter and end, make boolean array
    # if the FoQ > 0.05 True, if not False
    DTS_boolean = FoQ_raw > FoQ_threshold

    print("select HMM label csv file")
    HMM_label_path = tkinter.filedialog.askopenfilename()
    HMM_label = pd.read_csv(HMM_label_path)["state"]
    HMM_label = pd.concat([timeaxis, HMM_label], axis=1)
    HMM_label.columns = ["time_axis(min)", "state"]
    HMM_label = HMM_label.set_index("time_axis(min)")

    # Lethargus analysis
    DTS_analysis_with_HMM_label(analysis_res_df,
                                FoQ_raw,
                                DTS_boolean,
                                HMM_label,
                                body_size)

if __name__ == '__main__':
    main()