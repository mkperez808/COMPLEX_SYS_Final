# Matthew Perez

. ./path.sh
stage=3
ASR_stage=1

label_type="late"
#label_type="late manifest"
data_type="train val"
late="47647 81392 44574 35063 18771 76373 45758 68117 95407 56896 26753 47939 80292 58812"

early_late="25066 44574 18771 95465 13691 82697 35063 42080 32762 76373 45758 38717 68117 00359 \
81392 73828 95407 56896 05068 26753 29735 53870 47647 24371 47939 50377 78597 16486 80292 55029 \
58812 44209 07920 18261 11739 33752 61496 88947" #38

# if [ $stage -le 1 ]; then
#     for label in $label_type; do
#         if [ ! -d $SS_feats ]; then
#             mkdir -p $SS_feats
#         fi
#         echo "VAD feature extraction"
#         python $helper_path/VAD_extraction.py $raw_SS_wav $SS_feats $label
#     done
# fi

if [ $stage -le 1 ]; then
    echo "Segment MPT"
    python $helper_path/segment_MPT.py $raw_SS_wav $SS_Phonation
fi
# exit

if [ $stage -le 2 ]; then
    echo "Phonation features"
    python $helper_path/Phonation_extraction.py $SS_Phonation $SS_feats $helper_path/openSMILE_prosodic.conf
fi
# exit

# if  [ $stage -le 3 ]; then
#     for audio in $SS_Phonation/seg_wavs/*.wav; do
#         echo "Extract pitch and amplitude with kaldi"

#         full_wav_path="$audio"
#         base=`basename $audio`
#         file_only="${base%.*}"

#         echo "$file_only $full_wav_path" >> $SS_Phonation/seg_wavs/wav.scp
#         echo "$file_only $file_only" >> $SS_Phonation/seg_wavs/utt2spk
#     done
#     utils/utt2spk_to_spk2utt.pl $SS_Phonation/seg_wavs/utt2spk > $SS_Phonation/seg_wavs/spk2utt

#     #validate dir
#     utils/validate_data_dir.sh $SS_Phonation/seg_wavs
#     utils/fix_data_dir.sh $SS_Phonation/seg_wavs
# fi

# if [ $stage -le 4 ]; then
#     echo "Make feats.scp"
#     steps/make_mfcc.sh --nj 2 --mfcc-config $helper_path/mfcc.conf --cmd utils/run.pl $SS_Phonation/seg_wavs $SS_Phonation/seg_wavs/mfcc/log $SS_Phonation/seg_wavs/mfcc

# fi

# if [ $stage -le 5 ]; then
#     echo "Compute Pitch and Energy"
#     # Pitch and energy computation
#     compute-kaldi-pitch-feats --sample-frequency=32000 "scp:$SS_Phonation/seg_wavs/wav.scp" "ark,t:$SS_Phonation/pitch.txt"
#     copy-feats "scp:$SS_Phonation/seg_wavs/feats.scp" "ark,t:$SS_Phonation/energy.txt"
# fi
#exit


if [ $ASR_stage -le 1 ]; then
    if [ ! -d $SS_results ]; then
        mkdir -p $SS_results
    fi
    echo "Classification"
    python $helper_path/classify.py $SS_feats $label_type $SS_results
    # python $helper_path/classify_reg.py $SS_feats $label_type $SS_results $SS_Phonation
    # python $helper_path/classify_NN.py $SS_feats $label_type $SS_results
    # python $helper_path/sweep_DT_RF.py $SS_feats $label_type $SS_results
fi
exit

if [ $ASR_stage -le 2 ]; then
    echo "analyze features"
    if [ ! -d "$SS_results/feature_analysis" ]; then
        mkdir -p "$SS_results/feature_analysis"
    fi
    python $helper_path/analyze_features.py $SS_feats "$SS_results/feature_analysis" $SS_results

fi
