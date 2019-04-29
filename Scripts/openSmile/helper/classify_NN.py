import pandas as pd
import numpy as np
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import SGD
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, roc_auc_score, balanced_accuracy_score
from sklearn.feature_selection import SelectKBest, SelectPercentile, chi2, f_classif
from tqdm import tqdm
import operator
import os
import sys
from time import time

# TODO: output confusion matrix
# Log accuracies with respect to segment length

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

def Keras_NN(df, do):
    #return keras model

    #Get input shape size
    feature_size = df.shape[1]

    model = Sequential()
    # model.add(Dense(128, activation='relu', input_dim=feature_size))
    model.add(Dense(128, activation='relu', input_shape=(feature_size,)))
    model.add(Dropout(do))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(do))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(do))
    model.add(Dense(2, activation='softmax'))
    # model.add(Dense(1, activation='sigmoid'))

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def simulate_LOO(comb_df, label_type):
    # Create SKF
    acc_list = []
    uar_list = []
    auc_list = []
    feat_importance = {}
    conf_mat = {}
    tot_p = np.array([])
    tot_t = np.array([])

    #Get speakers
    # spkrs_list = set([i.split('-')[0] for i in comb_df.index.tolist()])
    spkrs_list = comb_df.index.tolist()

    y_true = []
    y_pred = []
    y_pred_prob = []
    len_feats=0

    #Split using LOO
    for te_spkr in spkrs_list:
        deep_df = comb_df.copy()

        #find te and tr rows
        te_utts = [i for i in deep_df.index.tolist() if i == te_spkr]
        tr_utts = [i for i in deep_df.index.tolist() if i != te_spkr]
        # te_utts = [i for i in deep_df.index.tolist() if i.split('-')[0] == te_spkr]
        # tr_utts = [i for i in deep_df.index.tolist() if i.split('-')[0] != te_spkr]

        # print(len(set([i.split('-')[0] for i in comb_df.index.tolist() if healthy_all_late[i.split('-')[0]] == 3])))
        # print(len(set([i.split('-')[0] for i in comb_df.index.tolist() if healthy_all_late[i.split('-')[0]] == 0])))
        # exit()
        #remove te/tr rows
        X_tr = deep_df.drop(te_utts)
        X_te = deep_df.drop(tr_utts)
        # print(X_tr['label'])

        #Balance training set -> Avoid overfitting due to class
        # error overtime
        # X_tr = balance_utts_df(X_tr)

        #extract labels
        Y_tr = X_tr['label']
        Y_te = X_te['label']
        # assert Y_tr.tolist().count(3) == Y_tr.tolist().count(0)
        # print(Y_tr)

        # Y_tr = np.array([1 if x>0 else 0 for x in Y_tr])
        # Y_te = np.array([1 if x>0 else 0 for x in Y_te])
        # print('Xte', Y_tr)
        # exit()
        # Y_tr

        # exit()

        #drop labels
        X_tr = X_tr.drop(columns=['label'])
        X_te = X_te.drop(columns=['label'])

        if len(X_te) == 0:
            continue

        # Z-normalization (make features on same scale). Important for NN
        for col in X_tr.columns.values:
           X_tr[col] = (X_tr[col] - X_tr[col].mean())/X_tr[col].std()
           X_te[col] = (X_te[col] - X_tr[col].mean())/X_tr[col].std()
        # exit()


        len_feats=X_tr.shape[1]

        # Classify using NN
        clf = Keras_NN(X_tr, 0.3)

        #callbacks
        es = EarlyStopping(monitor='val_loss', patience=150)
        mc = ModelCheckpoint('best_model.h5', monitor='val_acc', save_best_only=True)
        tensorboard = TensorBoard(log_dir='tensorboard_logs/{}'.format(time()))

        #stratified split  -> validation
        # xprint('pre', X_tr.shape)
        print(Y_tr.shape)

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15)
        for tr_idx, va_idx in sss.split(X_tr, Y_tr):
            X_tr, X_va = X_tr.iloc[tr_idx], X_tr.iloc[va_idx]
            Y_tr, Y_va = Y_tr.iloc[tr_idx], Y_tr.iloc[va_idx]
            
        # print('post', X_tr.shape, Y_tr.shape)
        # print('post', X_va.shape)
        # exit()
        # print('train', len(X_tr))
        # print('validation', len(X_va))
        # print('validation y', Y_va.tolist())
        # exit()

        Y_tr = keras.utils.to_categorical(Y_tr, num_classes=2)
        Y_va = keras.utils.to_categorical(Y_va, num_classes=2)
        Y_te = keras.utils.to_categorical(Y_te, num_classes=2)





        history = clf.fit(X_tr, Y_tr, validation_data=(X_va, Y_va), epochs=250, batch_size=1, callbacks=[es,mc, tensorboard], verbose=1)

        saved_model = load_model('best_model.h5')
        predictions = saved_model.predict_classes(X_te)
        predictions_prob = saved_model.predict(X_te)
        # print(predictions)
        # exit()

        # print('true', Y_te.tolist())
        # print('pred', predictions)
        # print('prob', predictions_prob)
        # exit()
        acc = accuracy_score(Y_te, predictions)
        # uar = recall_score(Y_te, predictions, average='binary')
        # auc = roc_auc_score(Y_te, predictions_prob)
        #uar_length = output_length_accuracies(X_te, predictions, Y_te)
        #roc = roc_auc_score(Y_te, predict_probs, average='weighted')
        acc_list.append(acc)
        # uar_list.append(uar)
        # auc_list.append(auc)

        # for feat, importance in zip(X_tr, clf.feature_importances_):
        #     if feat not in feat_importance.keys():
        #         feat_importance[feat] = []
        #     feat_importance[feat].append(importance)

        y_pred+=predictions.tolist()
        y_pred_prob+=predictions_prob.tolist()
        y_true+=Y_te.tolist()

    return np.mean(acc_list), len_feats, y_true, y_pred, y_pred_prob

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
    # # print('new_count',new_count)
    # # exit()
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

    combined_csv = os.path.join(feat_path,'openSMILE','comb','134_all.csv')
    pause_csv = os.path.join(feat_path, 'late_pause_feats.csv')

    os.chdir('/z/mkperez/SpontSpeech-HD/Scripts/openSmile')
    # print(os.getcwd())
    # exit()
    # Read data
    main_df = pd.read_csv(combined_csv, index_col=0)
    pause_df = pd.read_csv(pause_csv, index_col=0)

    # add pause(vad) feature_selection
    main_df = pd.concat([main_df, pause_df], axis=1, sort=True)

    for param in ['25']:
    # for param in ['25','50','75']:
        phonation_csv = os.path.join(feat_path, param+'_phonation_no_vad_first_break','comb_feats_'+param+'.csv')

        # Read data
        main_df = pd.read_csv(combined_csv, index_col=0)
        pause_df = pd.read_csv(pause_csv, index_col=0)
        phonation_df = pd.read_csv(phonation_csv, index_col=0)

        # add pause(vad) features
        main_df = pd.concat([main_df, pause_df], axis=1, sort=True)

        # USE PHONATION only
        main_df = phonation_df
        main_df.index = [pad_zeroes(i) for i in main_df.index.tolist()]

        # # Edit check microphone influence
        # drop_spkrs = [i for i in main_df.index.tolist() if early_late_balanced[i]==2]
        # main_df = main_df.drop(drop_spkrs, axis=0)

        #Add Labels
        # Binary labels
        main_df['label'] = [early_late_balanced[i] if early_late_balanced[i]==0 else 1 for i in main_df.index.tolist()]
        # main_df['label'] = [early_late_balanced[pad_zeroes(i)] for i in main_df.index.tolist()]

        #simulations
        # for length in [100, 250, 500, 750]:
        for length in [250]:
            # for feat_type in ["all", "Pause", "Prosodic"]:
            for feat_type in ["all"]:
                acc_list = []
                uar_list = []
                auc_list = []
                tot_importance = {}
                y_true =[]
                y_pred = []
                y_pred_prob=[]
                for i in tqdm(range(1)):
                    df = main_df.copy() #deep copy

                    #filter dataframe based on length (train and test with various segment lengths)
                    # dur_key = 'F0_sma_duration'
                    # # drop_rows = [i for i in df['voiceProb_sma_duration'].index if df['voiceProb_sma_duration'][i] < length]
                    # # df = df.drop(drop_rows)

                    # #drop other durations not dur_key
                    # drop_duration = [i for i in df.columns.values if 'duration' in i and i != dur_key]
                    # df = df.drop(drop_duration, axis=1)
                    
                    # drop_other = [i for i in df.columns.values if 'F0raw' in i or 'nnz' in i or 'frameTime' in i]
                    # df = df.drop(drop_other, axis=1)

                    # #drop features which have no std across speakers
                    drop_std_feats = [i for i in df if np.std(df[i]) == 0]
                    df = df.drop(drop_std_feats, axis=1)

                    # # Feature Selection - Types of feats
                    # if feat_type == "Pause":
                    #     pause_feats = [x for x in df.columns.values if 'pause' in x or 'vcd' in x or 'label' in x]
                    #     df = df[pause_feats]
                    # elif feat_type == "Prosodic":
                    #     prosodic_feats = [x for x in df.columns.values if 'pause' not in x and 'vcd' not in x]
                    #     df = df[prosodic_feats]

                    # #Add balanced number of healthy utts and HD utts
                    df = balance_utts_df(df)

                    #remove old best model
                    if os.path.exists('best_model.h5'):
                        os.remove('best_model.h5')

                    #Simulation!
                    # acc, uar, auc, len_feats, y_t, y_p = simulate_LOO(df, label_type)
                    acc, len_feats, y_t, y_p, y_pp = simulate_LOO(df, label_type)
                    y_true+=y_t
                    y_pred+=y_p
                    y_pred_prob+=y_pp
                    acc_list.append(accuracy_score(y_true, y_pred))
                    uar_list.append(recall_score(y_true, y_pred, average='binary'))
                    auc_list.append(roc_auc_score(y_true, y_pred_prob, average='weighted'))

     
                # Write results to file
                if not os.path.exists(os.path.join(results_base_path,'phonation')):
                    os.makedirs(os.path.join(results_base_path,'phonation'))
                results_file = os.path.join(results_base_path,'phonation', 'NN_binary_no_vad_first_break'+param+'_'+str(len_feats)+'.txt')

                with open(results_file,'w') as w:
                    w.write("Confusion Matrix\n")
                    # w.write(confusion_matrix(y_true, y_pred).tolist())
                    w.write('{}\n\n'.format(confusion_matrix(y_true, y_pred)))

                    w.write('Acc: {} ({})\n'.format(np.mean(acc_list), np.std(acc_list)))
                    w.write('UAR: {} ({})\n'.format(np.mean(uar_list), np.std(uar_list)))
                    w.write('Auc: {} ({})\n'.format(np.mean(auc_list), np.std(auc_list)))
                    



if __name__ == '__main__':
    main()