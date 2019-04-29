import sys
import os
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import numpy as np

def pad_zeroes(x):
    # take in integer, return string padded to 5 (w/ 0's in front)
    #speaker based
    new_num = str(x)
    while len(new_num) < 5:
        new_num="0"+new_num

    return str(new_num)


def main():
    feat_path = sys.argv[1]
    feat_analysis_path = sys.argv[2]
    results_path = sys.argv[3]

    for param in ['DT', 'RF']:
        csv = os.path.join(results_path, 'phn_sweep',param+'_sweep_results.csv')
        #Plot parameter sweep results
        results_df = pd.read_csv(csv, index_col=0)
        plt.figure()
        # sns_plot = sns.barplot(x="label", y=column, data=df)
        cols = [x for x in results_df.columns.values]
        # print(cols)
        # exit()

        if param == 'RF':
            results_df = pd.melt(results_df.reset_index(), id_vars='index', value_vars=cols)
            # print('post', results_df)
            sns_plot = sns.scatterplot(x='index', y='variable', hue='value', data=results_df, legend='brief')
            axes = sns_plot.axes
            axes.set_xticks([2,4,6,8])
            sns_plot.set(xlabel='Max Depth', ylabel='Number of Trees (Agents)', title="Random Forest Parameter Sweep")
            axes.get_legend().set_title("Accuracy")
            sns_plot.get_figure().savefig(os.path.join(results_path, param+"_plot.pdf"))
            plt.close()
        else:
            # print(results_df)
            mean_per_max_d = [np.mean(row.tolist()) for i, row in results_df.iterrows()]
            std_per_max_d = [np.std(row.tolist()) for i, row in results_df.iterrows()]
            results_df = results_df[[]]
            results_df['mean'] = mean_per_max_d
            results_df['std'] = std_per_max_d
            # print(results_df['mean'])
            # print(results_df['std'])
            # exit()
            results_df = pd.melt(results_df.reset_index(), id_vars='index', value_vars=cols)
            # print(results_df)
            # print(results_df.columns.values)
            # print(results_df['index'])
            # exit()
            sns_plot = sns.barplot(x='index', y='value', data=results_df)
            axes = sns_plot.axes
            axes.set_ylim(0.6, 0.7)
            sns_plot.set(xlabel='Max Depth', ylabel='Accuracy', title="Decision Tree Parameter Sweep")
            sns_plot.get_figure().savefig(os.path.join(results_path, param+"_plot.pdf"))
            plt.close()
    exit()

    #Plot all features
    for param in ['25', '50', '75']:
        # Main Data
        df = pd.read_csv(os.path.join(feat_path, param+'_phonation_no_vad_first_break', 'comb_feats_'+param+'.csv'), index_col=0)
        # print(df.index.tolist())
        # exit()
        # Read in labels data
        reg_csv = os.path.join(feat_path, 'HD_labels.csv')
        regre_labels = pd.read_csv(reg_csv, index_col=0)

        # Create labels
        reg_dic = {}
        reg_index_lst = [i[3:] for i in regre_labels.index.tolist()]
        UHDRS = [i for i in regre_labels['UHDRS_TFC_SUM']]
        DYS = [i for i in regre_labels['Dysarthria']]
        stage = [i for i in regre_labels['Staging']]
        for spkr, uh_label, d_label, st in zip(reg_index_lst, UHDRS, DYS, stage):
            reg_dic[spkr] = (uh_label, d_label, st)


        #add labels
        # print(df.index.tolist())
        # exit()
        stage_dic = {'Control': 0, 'Early':1, 'Late':2}
        df.index = [pad_zeroes(i) for i in df.index.tolist()]
        reg_list = [reg_dic[i][0] for i in df.index.tolist()]
        dys_list = [reg_dic[i][1] for i in df.index.tolist()]
        stage_list = [stage_dic[reg_dic[i][2]] for i in df.index.tolist()]
        df['reg_label'] = reg_list
        df['dys_label'] = dys_list
        df['label'] = stage_list

                

        # DF edit
        #drop all 'durations' and 'nnz' but 1
        dur_key = 'F0_sma_duration'
        drop_duration = [i for i in df.columns.values if 'duration' in i and i != dur_key]
        df = df.drop(drop_duration, axis=1)
        nnz_key = 'F0_sma_nnz'
        drop_nnz = [i for i in df.columns.values if 'nnz' in i and i != nnz_key]
        df = df.drop(drop_nnz, axis=1)


        #go through features
        no_labels = [i for i in df.columns.values if 'label' not in i]
        for column in tqdm(no_labels):
            col_short = column.replace('_sma', '')
            plt.figure()
            sns_plot = sns.barplot(x="label", y=column, data=df)
            sns_plot.get_figure().savefig(os.path.join(feat_analysis_path, col_short+'_'+param+".pdf"))
            plt.close()



if __name__ == '__main__':
    main()