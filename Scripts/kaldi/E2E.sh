#Matthew Perez
#For End-to-End system
# VAD + speaker diarization
# openSMILE
# Predict UHDRS score (regression)

. ./path.sh
stage=2
label_type="late"
#label_type="late manifest"
data_type="train val"
train_cmd="run.pl"


########  KALDI PRE-SETUP  ##################

if [ $stage -le 0 ]; then
    echo "Create segments"
    if [ ! -d $EE_kaldi/foo ]; then
        mkdir -p $EE_kaldi/foo/wav.scp
    fi

    #segments
    for audio in $raw_SS/*.wav; do
        full_wav_path="$audio"
        base=`basename $audio`
        file_only="${base%.*}"

        # Wav.scp
        # echo "$file_only $full_wav_path -t wav -c 1 - remix 2 |" >> $EE_kaldi/foo/wav.scp
        echo "$file_only $full_wav_path" >> $EE_kaldi/foo/wav.scp
        # Utt2spk
        echo "$file_only $file_only" >> $EE_kaldi/foo/utt2spk
    done

    #Create spk2utt for whole spkr
    utils/utt2spk_to_spk2utt.pl $EE_kaldi/foo/utt2spk > $EE_kaldi/foo/spk2utt

    #validate dir
    utils/validate_data_dir.sh $EE_kaldi/foo
    utils/fix_data_dir.sh $EE_kaldi/foo
fi

#exit
if [ $stage -le 1 ]; then
    echo "Make MFCC"
    steps/make_mfcc.sh --nj 2 --mfcc-config $helper_path/mfcc.conf --cmd utils/run.pl $EE_kaldi/foo $EE_kaldi/foo/mfcc/log $EE_kaldi/foo/mfcc
    # exit
    steps/compute_cmvn_stats.sh $EE_kaldi/foo $EE_kaldi/foo/mfcc/log $EE_kaldi/foo/mfcc
fi

if [ $stage -le 2 ]; then
    echo "VAD segments"
    # Compute VAD.scp
    # steps/compute_vad_decision.sh $EE_kaldi/foo

    # Compute segments using VAD.scp
    steps/vad_to_segments.sh $EE_kaldi/foo $EE_kaldi/foo/VAD_segments
fi

exit

if [ $stage -le 1 ]; then
    for label in $label_type; do
        python $helper_path/SS_data_preprocessing.py $raw_SS $SS_kaldi "true" $label
        exit
    done
fi
exit