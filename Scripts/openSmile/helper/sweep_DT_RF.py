import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, roc_auc_score
from sklearn.feature_selection import SelectKBest, SelectPercentile, chi2, f_classif
from tqdm import tqdm
import operator
import os
import sys


healthy_all_late = {'46352': 0, '95465': 0, '13691': 0, '82975': 0, '35063': 3, '29758': 0, 
'87083': 0, '71834': 0, '76373': 3, '78080': 0, '68117': 0, '81392': 3, '95407': 3, '56896': 3, '75256': 0, 
'88947': 0, '47647': 0, '25066': 0, '52053': 0, '16486': 0, '18771': 0, '45758': 0, '91117': 0, '26753': 0, 
'29735': 0, '61496': 0, '23789': 0, '47939': 0, '05068': 0, '80292': 3, '58812': 0, '44209': 0, '07920': 0, 
'44574': 3, '11739': 0}

healthy_premanifest = {'46352': 0, '20221': 1, '78971': 1, '52053': 0, '82975': 0, '69573': 1, 
'23789': 0, '75256': 0, '76883': 1, '29758': 0, '52080': 1, '78080': 0, '69695': 1, '01634': 1, 
'91117': 0, '59826': 1, '87083': 0, '14630': 1, '84596': 1, '71834': 0}

early_late_balanced = {'44574': 3, '95465': 0, '13691': 0, '82697': 0, '35063': 3, '11739': 0, '76373': 3,
'38717': 2, '68117': 0, '00359': 2, '81392': 3, '07920': 0, '95407': 3, '56896': 3, '24371': 2, '78597': 2,
'18261': 2, '53870': 2, '61496': 0, '88947': 0, '47647': 0, '32762': 2, '25066': 0, '16486': 0, '18771': 0,
'45758': 0, '50377': 2, '26753': 0, '29735': 0, '73828': 2, '47939': 0, '80292': 3, '55029': 2, '58812': 0,
'44209': 0, '42080': 2, '05068': 0, '33752': 2}

late_balanced = {'47647': 0, '35063': 3, '56896': 3,'26753': 0, '44574': 3, '81392': 3, '18771': 0,
'76373': 3, '45758': 0, '47939': 0, '80292': 3, '58812': 0, '68117': 0, '95407': 3}

def simulate_LOO(comb_df, param, max_d, num_estimators):
    # Create SKF
    acc_list = []
    feat_importance = {}
    conf_mat = {}
    tot_p = np.array([])
    tot_t = np.array([])

    #Get speakers
    # spkrs_list = set([i.split('-')[0] for i in comb_df.index.tolist()])
    spkrs_list = comb_df.index.tolist()

    y_true = []
    y_pred = []
    len_feats=0

    #Split using LOO
    for te_spkr in spkrs_list:

        #find te and tr rows
        # te_utts = [i for i in comb_df.index.tolist() if i.split('-')[0] == te_spkr]
        # tr_utts = [i for i in comb_df.index.tolist() if i.split('-')[0] != te_spkr]
        te_utts = [i for i in comb_df.index.tolist() if i == te_spkr]
        tr_utts = [i for i in comb_df.index.tolist() if i != te_spkr]

        #remove te/tr rows
        X_tr = comb_df.drop(te_utts)
        X_te = comb_df.drop(tr_utts)

        #Balance training set -> Avoid overfitting due to class
        # X_tr = balance_utts_df(X_tr)

        #extract labels
        Y_tr = X_tr['label']
        Y_te = X_te['label']
        # assert Y_tr.tolist().count(3) == Y_tr.tolist().count(0)
        # exit()

        #drop labels
        X_tr = X_tr.drop(columns=['label'])
        X_te = X_te.drop(columns=['label'])

        if len(X_te) == 0:
            continue

        # Filter features
        # filtered_feats = feature_selection(X_tr, Y_tr, percent=80)
        # len_feats=len(filtered_feats)
        # X_tr = X_tr[filtered_feats]
        # X_te = X_te[filtered_feats]

        len_feats=X_tr.shape[1]

        # Classify using randomForest
        if param == 'RF':
            clf = RandomForestClassifier(n_estimators=num_estimators, max_depth=max_d)
        else:
            clf = DecisionTreeClassifier(max_depth=max_d)
        clf.fit(X_tr, Y_tr)
        predictions = clf.predict(X_te)
        # print(Y_te.index, Y_te.tolist(), predictions)
        #print(predictions)
        #exit()

        # predict_probs = clf.predict_proba(X_te)

        acc = accuracy_score(Y_te, predictions)
        uar = recall_score(Y_te, predictions, average='macro')
        #uar_length = output_length_accuracies(X_te, predictions, Y_te)
        #roc = roc_auc_score(Y_te, predict_probs, average='weighted')
        acc_list.append(acc)

        for feat, importance in zip(X_tr, clf.feature_importances_):
            if feat not in feat_importance.keys():
                feat_importance[feat] = []
            feat_importance[feat].append(importance)

        y_pred+=predictions.tolist()
        y_true+=Y_te.tolist()

    return np.mean(acc_list), feat_importance, len_feats, y_true, y_pred

def feature_selection(X_tr, Y_tr, percent):
    # Run feature selection
    # Return list of top features to use
    selector = SelectPercentile(f_classif, percentile=percent)
    X_new = selector.fit_transform(X_tr, Y_tr)

    filtered_feats = [column[0] for column in zip(X_tr.columns,selector.get_support()) if column[1]]

    # print(X_tr.shape[1], len(filtered_feats))
    # exit()
    return filtered_feats

def balance_utts_df(df):
    #Balance number of healthy and HD utterances
    #TODO: Change for multi-class
    labels_arr = df['label']
    classes = set(labels_arr)

    count = {}
    for i in classes:
        count[i] = 0
    for i in labels_arr:
        count[i]+=1
    min_label = min(count, key=count.get)
    min_label_count = count[min_label]


    for i in classes: # go through all label types
        if count[i] > min_label_count:
            #filter
            num_filter = count[i] - min_label_count
            all_filtered_labels = [x for x, row in df.iterrows() if row['label'] == i]
            filtered_rows = np.random.choice(all_filtered_labels, size=num_filter, replace=False)
            # print(filtered_rows)
            # exit()
            # print(num_filter, len(filtered_rows))
            # print('pre drop', len(df.index.tolist()))
            df = df.drop(filtered_rows)

    # print(df['label'])
    # exit()
    # sanity check 
    new_classes = set(df['label'])
    new_count = {}
    for i in new_classes:
        new_count[i] = 0
    for i in df['label']:
        new_count[i]+=1
    # print('new_count',new_count)
    # exit()
    return df


def pad_zeroes(x):
    # take in integer, return string padded to 5 (w/ 0's in front)
    #speaker based
    new_num = str(x)
    while len(new_num) < 5:
        new_num="0"+new_num

    return str(new_num)

def main():
    #Read in csv file
    feat_path = sys.argv[1]
    label_type = sys.argv[2]
    results_base_path = sys.argv[3]



    num_estimators_sweep = [5, 10, 50, 100, 250, 500, 1000]
    max_depth_sweep = [2, 4, 6, 8]
    # Phonation
    for param in ['DT', 'RF']:
        results = {}
        for max_d in max_depth_sweep:
            results[max_d] = []
            for num_e in num_estimators_sweep:
                phonation_csv = os.path.join(feat_path,'25_phonation_no_vad_first_break','comb_feats_25.csv')

                # Read data
                phonation_df = pd.read_csv(phonation_csv, index_col=0)

                # add pause(vad) features
                # main_df = pd.concat([main_df, pause_df], axis=1, sort=True)

                # USE PHONATION only
                main_df = phonation_df
                main_df.index = [pad_zeroes(i) for i in main_df.index.tolist()]
                # print(main_df.index)
                # exit()

                # # Edit check microphone influence
                # drop_spkrs = [i for i in main_df.index.tolist() if early_late_balanced[i]==2]
                # main_df = main_df.drop(drop_spkrs, axis=0)

                #Add Labels
                # Binary labels
                main_df['label'] = [early_late_balanced[i] if early_late_balanced[i]==0 else 1 for i in main_df.index.tolist()]
                # main_df['label'] = [early_late_balanced[pad_zeroes(i)] for i in main_df.index.tolist()]
                acc_list = []
                tot_importance = {}
                y_true =[]
                y_pred = []
                for i in tqdm(range(10)):
                    df = main_df.copy() #deep copy

                    #filter dataframe based on length (train and test with various segment lengths)
                    dur_key = 'F0_sma_duration'
                    nnz_key = 'F0_sma_nnz'
                    # drop_rows = [i for i in df['voiceProb_sma_duration'].index if df['voiceProb_sma_duration'][i] < length]
                    # df = df.drop(drop_rows)

                    #drop other durations not dur_key
                    drop_duration = [i for i in df.columns.values if 'duration' in i and i != dur_key]
                    df = df.drop(drop_duration, axis=1)
                    drop_nnz = [i for i in df.columns.values if 'nnz' in i and i != nnz_key]
                    df = df.drop(drop_nnz, axis=1)
                    
                    drop_other = [i for i in df.columns.values if 'F0raw' in i or 'frameTime' in i or 'max' in i or 'min' in i or 'mean' in i]
                    df = df.drop(drop_other, axis=1)
                    keep = [i for i in df.columns.values if 'intensity' in i or 'F0' in i or 'label' in i]
                    df = df[keep]
                    # print(df.columns.values, len(df.columns.values))
                    # exit()

                    # #drop features which have no std across speakers
                    drop_std_feats = [i for i in df if np.std(df[i]) == 0]
                    df = df.drop(drop_std_feats, axis=1)

                    # #Add balanced number of healthy utts and HD utts
                    df = balance_utts_df(df)

                    #Simulation!
                    acc, importance, len_feats, y_t, y_p = simulate_LOO(df, param, max_d, num_e)
                    # acc, importance, len_feats, y_t, y_p = simulate_kFold(df, label_type, 10)
                    acc_list.append(acc)
                    y_true+=y_t
                    y_pred+=y_p

                    # print(len_feats)
                    # exit()
                    for key in importance:
                        if key not in tot_importance:
                            tot_importance[key] = []
                        tot_importance[key] += importance[key]
     
                # Write results to file
                if not os.path.exists(os.path.join(results_base_path,'phn_sweep')):
                    os.makedirs(os.path.join(results_base_path,'phn_sweep'))
                results_file = os.path.join(results_base_path,'phn_sweep', param+'_binary_no_vad_first_break_'+str(len_feats)+'.txt')

                with open(results_file,'w') as w:
                    w.write("Confusion Matrix\n")
                    # w.write(confusion_matrix(y_true, y_pred).tolist())
                    w.write('{}\n\n'.format(confusion_matrix(y_true, y_pred)))

                    w.write('Acc: {} ({}). acc_list {}\n'.format(np.mean(acc_list), np.std(acc_list), acc_list))
                    w.write("\nFeature Importance\n")
                    for i in tot_importance:
                        tot_importance[i] = np.mean(tot_importance[i])
                    for i in sorted(tot_importance.items(), key=operator.itemgetter(1), reverse=True):
                        w.write("{} = {}\n".format(i[0], i[1]))

                #save result
                results[max_d].append(np.mean(acc_list))
        result_df = pd.DataFrame.from_dict(results, orient='index', columns=num_estimators_sweep)
        result_df.to_csv(os.path.join(results_base_path,'phn_sweep',param+'_sweep_results.csv'))
        # exit()




if __name__ == '__main__':
    main()