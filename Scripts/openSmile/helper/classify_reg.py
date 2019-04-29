import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, ShuffleSplit
from sklearn.metrics import confusion_matrix, r2_score, balanced_accuracy_score, explained_variance_score, mean_squared_error
from sklearn.feature_selection import SelectKBest, SelectPercentile, chi2, f_classif
from tqdm import tqdm
import operator
import os
import sys
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt


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

def read_pitch_energy(pitch_path, energy_path):
    # get dictionary of pitch and energy for each speaker
    pitch={}
    spkr = 0
    with open(pitch_path, 'r') as r:
        for line in r:
            line = line.strip()

            if line.split()[1] == "[":
                # spkr = line.split()[0]
                # pitch[str(spkr)] = []
                spkr = line.split('_')[0]
                if spkr not in pitch.keys():
                    pitch[str(spkr)] = []
            else:
                pitch_val = line.split()[1]
                pitch[spkr].append(float(pitch_val))

    energy={}
    spkr = 0
    with open(energy_path, 'r') as r:
        for line in r:
            line = line.strip()

            if line.split()[1] == "[":
                #check for new speaker
                spkr = line.split('_')[0]
                if spkr not in energy.keys():
                    energy[str(spkr)] = []
            else:
                energy_val = line.split()[0]
                energy[spkr].append(float(energy_val))
    return pitch, energy

def compute_shapelet_frame(lst, length, padding):
    #extract shapelet at frame level (list of lists)
    if padding == True:
        shapelet_list = []
        for index in range(0, len(lst), length):
            shapelet = lst[index:index+length]
            if len(shapelet) < length:
                diff = length - len(shapelet)
                shapelet+=[0.0 for i in range(diff)]
            shapelet_list.append(shapelet)

        return shapelet_list

    else:
        shapelet_list = []
        for index in range(0, len(lst), length):
            shapelet = lst[index:index+length]
            shapelet_list.append(shapelet)

        return shapelet_list

def classify_with_shapelets():
    from keras.optimizers import Adagrad
    from tslearn.datasets import CachedDatasets
    from tslearn.preprocessing import TimeSeriesScalerMinMax
    from tslearn.shapelets import ShapeletModel, grabocka_params_to_shapelet_size_dict

    feat_path = sys.argv[1]
    label_type = sys.argv[2]
    results_base_path = sys.argv[3]
    data_path = sys.argv[4]

    pitch_path = os.path.join(data_path,'pitch.txt')
    energy_path = os.path.join(data_path,'energy.txt')

    #Raw pitch and energy
    raw_pitch, raw_energy = read_pitch_energy(pitch_path, energy_path)


    # Tunable Parameters = shapelet length, threshold, shapelet redundancy value 
    # sweep shapelet length
    pitch_shapelet = {}
    energy_shapelet = {}
    for shapelet_len in [10, 25, 50]:
        for spkr in raw_pitch:
            # Compute shapelets from raw frames (i.e. no segmented info like where phone/word is)
            pitch_shapelet[spkr] = compute_shapelet_frame(raw_pitch[spkr], shapelet_len, True)
            # energy_shapelet[spkr] = compute_shapelet_frame(raw_energy[spkr], shapelet_len)

            # pitch_shapelet[spkr] = np.array(raw_pitch[spkr])
            # print(len(raw_pitch[spkr]))
            # exit()

        acc = []
        for sim in range(10):
            y_true = []
            y_pred = []
            for spkr in tqdm(late_balanced.keys()):
                test_spkr = [spkr]
                train_spkrs = late_balanced.keys()
                train_spkrs.remove(test_spkr[0])


                X_train = np.array([np.array(shapelet).reshape(shapelet_len, 1) for x in train_spkrs for shapelet in pitch_shapelet[x]])
                y_train = np.array([late_balanced[x] for x in train_spkrs for shapelet in pitch_shapelet[x]])

                X_test = np.array([np.array(shapelet).reshape(shapelet_len, 1) for x in test_spkr for shapelet in pitch_shapelet[x]])
                y_test = np.array([late_balanced[x] for x in test_spkr for shapelet in pitch_shapelet[x]])

                # print('train data', X_train.shape)
                # #print('train data first', X_train[0])
                # print('train label', y_train.shape)
                # exit()

                shapelet_sizes = grabocka_params_to_shapelet_size_dict(n_ts=X_train.shape[0],
                                                                       ts_sz=X_train.shape[1],
                                                                       n_classes=len(set(y_train)),
                                                                       l=0.1,
                                                                       r=2)



                shp_clf = ShapeletModel(n_shapelets_per_size=shapelet_sizes,
                                        optimizer=Adagrad(lr=.1),
                                        weight_regularizer=.01,
                                        max_iter=50,
                                        verbose_level=0)
                shp_clf.fit(X_train, y_train)

                predicted_locations = shp_clf.locate(X_test)

                print('predicted_locations.shape',predicted_locations.shape)
                # test_ts_id = 0
                # plt.figure()
                # plt.title("Example locations of shapelet matches (%d shapelets extracted)" % sum(shapelet_sizes.values()))
                # plt.plot(X_test[test_ts_id].ravel())
                # for idx_shp, shp in enumerate(shp_clf.shapelets_):
                #     t0 = predicted_locations[test_ts_id, idx_shp]
                #     plt.plot(np.arange(t0, t0 + len(shp)), shp, linewidth=2)

                # plt.tight_layout()
                # plt.savefig(test_ts_id+'_test.png', format='png')
                # exit()

                prediction = shp_clf.predict(X_test)
                prediction_prob = shp_clf.predict_proba(X_test)

                y_pred+=prediction.tolist()
                y_true+=y_test.tolist()


            ###After LOO
            # test_ts_id = 0
            # plt.figure()
            # plt.title("Example locations of shapelet matches (%d shapelets extracted)" % sum(shapelet_sizes.values()))
            # plt.plot(X_test[test_ts_id].ravel())
            # for idx_shp, shp in enumerate(shp_clf.shapelets_):
            #     t0 = predicted_locations[test_ts_id, idx_shp]
            #     plt.plot(np.arange(t0, t0 + len(shp)), shp, linewidth=2)

            # plt.tight_layout()
            # plt.savefig('test.png', format='png')

            local_acc = balanced_accuracy_score(y_true, y_pred)
            acc.append(local_acc)
        # print('acc', acc)
        # print('final acc', np.mean(acc))
        # print('final acc std', np.std(acc))

        if not os.path.exists(os.path.join(results_base_path,'regression')):
            os.makedirs(os.path.join(results_base_path,'regression'))
        results_file = os.path.join(results_base_path,'regression', 'shapelet_'+str(len_feats)+'.txt')

        with open(results_file, 'w') as w:
            # w.write("Confusion Matrix\n")
            # # w.write(confusion_matrix(y_true, y_pred).tolist())
            # w.write('{}\n\n'.format(confusion_matrix(y_true, y_pred)))

            w.write('regression: {} ({})\n'.format(np.mean(acc_list), np.std(acc_list)))
            w.write('baseline: {} ({})'.format(np.mean(acc_baseline), np.std(acc_baseline)))
            w.write("\nFeature Importance\n")
            for i in tot_importance:
                tot_importance[i] = np.mean(tot_importance[i])
            for i in sorted(tot_importance.items(), key=operator.itemgetter(1), reverse=True):
                w.write("{} = {}\n".format(i[0], i[1]))

def find_top_features(X_tr, Y_tr, rf):
    scores = {}
    names = X_tr.columns.values
    #crossvalidate the scores on a number of different random splits of the data
    ss = ShuffleSplit(n_splits=10, test_size=.3)
    for train_idx, test_idx in ss.split(X_tr):
        X_train, X_test = X_tr.iloc[train_idx], X_tr.iloc[test_idx]
        Y_train, Y_test = Y_tr.iloc[train_idx], Y_tr.iloc[test_idx]
        r = rf.fit(X_train, Y_train)
        acc = r2_score(Y_test, rf.predict(X_test))
        for i in range(X_tr.shape[1]):
            X_t = X_test.copy()
            np.random.shuffle(X_t.iloc[:, i])
            shuff_acc = r2_score(Y_test, rf.predict(X_t))
            if names[i] not in scores:
                scores[names[i]] = []
            scores[names[i]].append((acc-shuff_acc)/acc)
    print "Features sorted by their score:"
    # print sorted([(round(np.mean(score), 4), feat) for
    #               feat, score in scores.items()], reverse=True)

    #select only positive features
    ranks = sorted([(round(np.mean(score), 4), feat) for
                  feat, score in scores.items()], reverse=True)
    top_feats = [x[1] for x in ranks if x[0]>0]
    # print(ranks)
    # print(top_feats)
    return top_feats
    exit()

def simulate_LOO(comb_df, label_type, results_base_path):
    # Create SKF
    acc_list = []
    acc_baseline = []
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
        te_utts = [i for i in comb_df.index.tolist() if i == te_spkr]
        tr_utts = [i for i in comb_df.index.tolist() if i != te_spkr]
        # te_utts = [i for i in comb_df.index.tolist() if i.split('-')[0] == te_spkr]
        # tr_utts = [i for i in comb_df.index.tolist() if i.split('-')[0] != te_spkr]

        #remove te/tr rows
        X_tr = comb_df.drop(te_utts)
        X_te = comb_df.drop(tr_utts)

        # print(X_tr.shape, X_te.index)
        #Balance training set -> Avoid overfitting due to class
        # X_tr = balance_utts_df(X_tr, 'label')

        #extract labels
        Y_tr = X_tr['reg_label']
        Y_te = X_te['reg_label']

        #drop labels
        X_tr = X_tr.drop(columns=['label', 'reg_label'])
        X_te = X_te.drop(columns=['label', 'reg_label'])

        if len(X_te) == 0:
            continue

        # Classify using randomForest
        clf = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=2)

        # Filter features
        # print('pre', X_tr.shape)
        # feats = find_top_features(X_tr, Y_tr, clf)
        # X_tr = X_tr[feats]
        # X_te = X_te[feats]
        # print('post', X_tr.shape)
        len_feats=X_tr.shape[1]

        clf.fit(X_tr, Y_tr)
        predictions = clf.predict(X_te)
        # print(predictions)
        # exit()

        # acc = accuracy_score(Y_te, predictions)
        # uar = recall_score(Y_te, predictions, average='macro')
        evs = mean_squared_error(Y_te, predictions)
        evs_baseline = mean_squared_error(Y_te, [np.mean(Y_tr) for i in range(len(Y_te))])
        # print(predictions, Y_te.tolist(), evs)
        # exit()


        #uar_length = output_length_accuracies(X_te, predictions, Y_te)
        #roc = roc_auc_score(Y_te, predict_probs, average='weighted')
        acc_list.append(evs)
        acc_baseline.append(evs_baseline)

        for feat, importance in zip(X_tr, clf.feature_importances_):
            if feat not in feat_importance.keys():
                feat_importance[feat] = []
            feat_importance[feat].append(importance)

        y_pred+=predictions.tolist()
        y_true+=Y_te.tolist()
    print(len_feats)
    # Plot the results
    f = plt.figure()
    s = 50
    a = 0.4
    plt.scatter([i for i in range(len(y_true))], y_true, edgecolor='k',
                c="navy", s=s, marker=".", alpha=a)

    plt.scatter([i for i in range(len(y_pred))], y_pred, edgecolor='k',
                c="red", s=s, marker="x", alpha=a)

    plt.xlim([0, len(y_true)])
    plt.ylim([0, 13])
    plt.xlabel("Samples")
    plt.ylabel("Target")
    plt.title("Measuring HD Dysarthria")
    plt.show()
    f.savefig(os.path.join(results_base_path,'regression',str(len_feats)+'_graph.pdf'))
    exit()

    return np.mean(acc_list), feat_importance, len_feats, y_true, y_pred, np.mean(acc_baseline)

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

            df = df.drop(filtered_rows)


    # sanity check 
    # new_classes = set(df['label'])
    # new_count = {}
    # for i in new_classes:
    #     new_count[i] = 0
    # for i in df['label']:
    #     new_count[i]+=1
    # print('new_count',new_count)
    # exit()
    return df

def pad_zeroes(x):
    # take in integer, return string padded to 5 (w/ 0's in front)
    new_num = str(x)
    while len(new_num) < 5:
        new_num="0"+new_num

    return str(new_num)

def main():
    #Read in csv file
    feat_path = sys.argv[1]
    label_type = sys.argv[2]
    results_base_path = sys.argv[3]
    combined_csv = os.path.join(feat_path,'openSMILE','comb','66_all.csv')
    pause_csv = os.path.join(feat_path, 'late_pause_feats.csv')
    reg_csv = os.path.join(feat_path, 'HD_labels.csv')

    # Read data
    main_df = pd.read_csv(combined_csv, index_col=0)
    pause_df = pd.read_csv(pause_csv, index_col=0)
    regre_labels = pd.read_csv(reg_csv, index_col=0)

    # Regression labels
    reg_dic = {}
    reg_index_lst = [i[3:] for i in regre_labels.index.tolist()]
    reg_UHDRS = [i for i in regre_labels['UHDRS_TFC_SUM']]
    reg_DYS = [i for i in regre_labels['Dysarthria']]
    for spkr, uh_label, d_label in zip(reg_index_lst, reg_UHDRS, reg_DYS):
        reg_dic[spkr] = (uh_label, d_label)

    # add pause(vad) features
    main_df = pd.concat([main_df, pause_df], axis=1, sort=True)

    #replace labels with regression labels
    utt_lst = main_df.index.tolist()
    reg_list = [reg_dic[i.split('-')[0]][0] for i in utt_lst ]
    dys_list = [reg_dic[i.split('-')[0]][1] for i in utt_lst ]
    #pd_reg_list = pd.Series(reg_list, name='label', index=utt_lst)
    
    # main_df['reg_label'] = reg_list
    # main_df['reg_label'] = dys_list
    # # main_df['reg_dys'] = dys_list
    # print(main_df.shape)
    # exit()

    for param in ['0_25', '0_50','0_75']:
        phonation_csv = os.path.join(feat_path,'phonation','comb_feats'+param+'.csv')
        phonation_df = pd.read_csv(phonation_csv, index_col=0)
        main_df = phonation_df
        main_df.index = [pad_zeroes(i) for i in main_df.index.tolist()]


        reg_list = [reg_dic[i][0] for i in main_df.index.tolist()]
        dys_list = [reg_dic[i][1] for i in main_df.index.tolist()]
        main_df['reg_label'] = reg_list
        # main_df['dys_label'] = dys_list
        # main_df['label'] = [early_late_balanced[i] for i in main_df.index.tolist()]
        # main_df.to_csv(os.path.join(feat_path, 'phonation_w_labels.csv'))
        # exit()
        main_df['label'] = [early_late_balanced[i] if early_late_balanced[i]==0 else 1 for i in main_df.index.tolist()]
        # print(main_df.shape)
        # exit()

        #simulations
        acc_list = []
        acc_baseline = []
        tot_importance = {}
        y_true =[]
        y_pred = []
        for i in tqdm(range(10)):
            df = main_df.copy() #deep copy

            # #filter dataframe based on length (train and test with various segment lengths)
            dur_key = 'F0_sma_duration'
            # drop_rows = [i for i in df['voiceProb_sma_duration'].index if df['voiceProb_sma_duration'][i] < length]
            # df = df.drop(drop_rows)

            #drop other durations not dur_key
            drop_duration = [i for i in df.columns.values if 'duration' in i and i != dur_key]
            df = df.drop(drop_duration, axis=1)
            
            drop_other = [i for i in df.columns.values if 'F0raw' in i or 'nnz' in i or 'frameTime' in i]
            df = df.drop(drop_other, axis=1)
            drop_other2 = [i for i in df.columns.values if 'LocEnv' in i or 'min' in i or 'max' in i]
            df = df.drop(drop_other2, axis=1)

            keep = [i for i in df.columns.values if 'F0_sma_linregerrA' in i or 'F0_sma_stddev' in i
                or 'pcm_intensity_sma_linregerrA' in i or 'pcm_intensity_sma_stddev' in i
                or 'label' in i]
            df = df[keep]

            # #drop features which have no std across speakers
            drop_std_feats = [i for i in df if np.std(df[i]) == 0]
            df = df.drop(drop_std_feats, axis=1)


            # print('pre balance', df.shape)
            # #Add balanced number of healthy utts and HD utts
            df = balance_utts_df(df)
            # print('post balance', df.shape)
            # exit()

            #Simulation!
            acc, importance, len_feats, y_t, y_p, acc_b = simulate_LOO(df, label_type, results_base_path)
            # acc, importance, len_feats, y_t, y_p = simulate_kFold(df, label_type, 10)
            acc_list.append(acc)
            acc_baseline.append(acc_b)
            y_true+=y_t
            y_pred+=y_p

            for key in importance:
                if key not in tot_importance:
                    tot_importance[key] = []
                tot_importance[key] += importance[key]
     
        # Write results to file
        # if not os.path.exists(os.path.join(results_base_path,'len_'+str(length))):
        #     os.makedirs(os.path.join(results_base_path,'len_'+str(length)))
        # results_file = os.path.join(results_base_path,'len_'+str(length), 'reg_'+param+'_'+str(len_feats)+'.txt')

        if not os.path.exists(os.path.join(results_base_path,'regression')):
            os.makedirs(os.path.join(results_base_path,'regression'))
        results_file = os.path.join(results_base_path,'regression', param+'_'+str(len_feats)+'.txt')

        with open(results_file, 'w') as w:
            # w.write("Confusion Matrix\n")
            # # w.write(confusion_matrix(y_true, y_pred).tolist())
            # w.write('{}\n\n'.format(confusion_matrix(y_true, y_pred)))

            w.write('regression: {} ({})\n'.format(np.mean(acc_list), np.std(acc_list)))
            w.write('baseline: {} ({})'.format(np.mean(acc_baseline), np.std(acc_baseline)))
            w.write("\nFeature Importance\n")
            for i in tot_importance:
                tot_importance[i] = np.mean(tot_importance[i])
            for i in sorted(tot_importance.items(), key=operator.itemgetter(1), reverse=True):
                w.write("{} = {}\n".format(i[0], i[1]))




if __name__ == '__main__':
    main()
    # classify_with_shapelets()

