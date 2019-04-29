#Matthew Perez
#For Spontaneous speech
. ./path.sh
stage=1
ASR_stage=1
label_type="manifest"
#label_type="late manifest"
data_type="train val"
train_cmd="run.pl"


########  KALDI PRE-SETUP  ##################
if [ $stage -le 1 ]; then
	for label in $label_type; do
		# For ASR
		python $helper_path/SS_data_preprocessing.py $raw_SS $SS_kaldi "false" $label

		# For speech processing
		# python $helper_path/kaldi_processing_acoustics.py $raw_SS $SS_kaldi "true" $label
		# exit
	done
fi
exit

if [ $stage -le 2 ]; then
	echo "Copying Dictionaries"
	$helper_path/SS_copy_dict.sh
fi


if [ $stage -le 3 ]; then
	echo "Spk2Utt prep"
	for sname in $data_type; do
		for data in $SS_kaldi/set*; do
			utils/utt2spk_to_spk2utt.pl $data/$sname/utt2spk > $data/$sname/spk2utt
		done
	done
	#All set
	utils/utt2spk_to_spk2utt.pl $SS_kaldi/all_data_test/test/utt2spk > $SS_kaldi/all_data_test/test/spk2utt
fi

if [ $stage -le 4 ]; then
	echo "Validate data directories"
	for sname in $data_type; do
		for data in $SS_kaldi/set*; do
			utils/validate_data_dir.sh $data/$sname
			utils/fix_data_dir.sh $data/$sname
			#exit
		done
	done
	#All set
	utils/validate_data_dir.sh $SS_kaldi/all_data_test/test
	utils/fix_data_dir.sh $SS_kaldi/all_data_test/test
fi
#exit
if [ $stage -le 5 ]; then
	echo "Feature creation (MFCC)"
	for sname in $data_type; do
		for data in $SS_kaldi/set*; do
			steps/make_mfcc.sh --nj 2 --mfcc-config $helper_path/mfcc.conf --cmd utils/run.pl $data/$sname $data/$sname/mfcc/log $data/$sname/mfcc
			#utils/fix_data_dir.sh $data/$sname
			#exit
			steps/compute_cmvn_stats.sh $data/$sname $data/$sname/mfcc/log $data/$sname/mfcc
		done
	done

	#All set
	steps/make_mfcc.sh --nj 2 --mfcc-config $helper_path/mfcc.conf --cmd utils/run.pl $SS_kaldi/all_data_test/test $SS_kaldi/all_data_test/test/mfcc/log $SS_kaldi/all_data_test/test/mfcc
	steps/compute_cmvn_stats.sh $SS_kaldi/all_data_test/test $SS_kaldi/all_data_test/test/mfcc/log $SS_kaldi/all_data_test/test/mfcc
fi

if [ $stage -le 6 ]; then
	echo "Generate bigram text"
	for data in $SS_kaldi/set*; do
		pushd $helper_path
		python gen-bigram-text.py $data/train/text $data/train/LM_text
		python gen-bigram-text.py $data/train/all_text_clinician $data/train/LM_all_text_clinician #incoporate all clinician data (including test clinician)
		popd
		#exit
	done
	#All data
	pushd $helper_path
	python gen-bigram-text.py $SS_kaldi/all_data_test/test/text $SS_kaldi/all_data_test/test/LM_text
	popd
fi


if [ $stage -le 7 ]; then
	echo "Prepare /lang directory"
	for data in $SS_kaldi/set*; do
		utils/prepare_lang.sh $data/dict '<UNK>' $data/local $data/lang
		utils/prepare_lang.sh $data/dict '<UNK>' $data/local_all_clinician $data/lang_all_clinician
		#exit
	done

	#All data
	utils/prepare_lang.sh $SS_kaldi/all_data_test/test/dict '<UNK>' $SS_kaldi/all_data_test/test/local $SS_kaldi/all_data_test/test/lang
fi

if [ $stage -le 8 ]; then
	echo "Create Langauge Model (G.fst)"
	for data in $SS_kaldi/set*; do
		$CHAI_SHARE_PATH/Bins/kaldi/build_lm.sh $data/train/LM_text $data/lang
		$CHAI_SHARE_PATH/Bins/kaldi/build_lm_order.sh $data/train/LM_all_text_clinician $data/lang_all_clinician 1
		#exit
	done

	#All data
	$CHAI_SHARE_PATH/Bins/kaldi/build_lm.sh $SS_kaldi/all_data_test/test/LM_text $SS_kaldi/all_data_test/test/lang
fi


################################################

############  ASR Training   ############


# mono='mono-2000'
# mono_ali=$mono'_ali'
if [ $ASR_stage -le 0 ]; then
	echo "Monophone training, alignment, decoding when using all data for training"
	steps/train_mono.sh --nj 12 --cmd "$train_cmd" $SS_kaldi/all_data_test/test $SS_kaldi/all_data_test/test/lang $SS_kaldi/all_data_test/test/exp/mono_all
	steps/align_si.sh --nj 12 --cmd "$train_cmd" $SS_kaldi/all_data_test/test $SS_kaldi/all_data_test/test/lang $SS_kaldi/all_data_test/test/exp/mono_all $SS_kaldi/all_data_test/test/exp/mono_all_ali

	graph_dir=$SS_kaldi/all_data_test/test/exp/mono_all_ali/graph
	utils/mkgraph.sh $SS_kaldi/all_data_test/test/lang $SS_kaldi/all_data_test/test/exp/mono_all_ali $graph_dir

	steps/decode.sh --nj 1 $graph_dir $SS_kaldi/all_data_test/test $SS_kaldi/all_data_test/test/exp/mono_all_ali/decode

		#steps/train_mono.sh --config $helper_path/train_mono.config --nj 12 --cmd utils/run.pl $fold/train $fold/lang $fold/exp/$mono
		#steps/align_si.sh --nj 12 --cmd utils/run.pl $fold/train $fold/lang $fold/exp/$mono $fold/exp/$mono_ali
fi

if [ $ASR_stage -le 1 ]; then
	for fold in $SS_kaldi/set*; do
		echo "Monophone training and alignment. Clinician data used for training language model"
		steps/train_mono.sh --nj 12 --cmd "$train_cmd" $fold/train $fold/lang_all_clinician $fold/exp/mono_clinician
		steps/align_si.sh --nj 12 --cmd "$train_cmd" $fold/train $fold/lang_all_clinician $fold/exp/mono_clinician $fold/exp/mono_ali_clinician

		graph_dir=$fold/exp/mono_ali_clinician/graph
		utils/mkgraph.sh $fold/lang_all_clinician $fold/exp/mono_ali_clinician $graph_dir

		steps/decode.sh --nj 1 $graph_dir $fold/val $fold/exp/mono_ali_clinician/decode
		#steps/train_mono.sh --config $helper_path/train_mono.config --nj 12 --cmd utils/run.pl $fold/train $fold/lang $fold/exp/$mono
		#steps/align_si.sh --nj 12 --cmd utils/run.pl $fold/train $fold/lang $fold/exp/$mono $fold/exp/$mono_ali
	done
fi
#exit

# if [ $ASR_stage -le 2 ]; then
# 	for fold in $SS_kaldi/set*; do
# 		echo "Monophone training and alignment. LOO w/ participant only"
# 		steps/train_mono.sh --nj 12 --cmd "$train_cmd" $fold/train $fold/lang $fold/exp/mono
# 		steps/align_si.sh --nj 12 --cmd "$train_cmd" $fold/train $fold/lang $fold/exp/mono $fold/exp/mono_ali

# 		graph_dir=$fold/exp/mono_ali/graph
# 		utils/mkgraph.sh $fold/lang $fold/exp/mono_ali $graph_dir

# 		steps/decode.sh --nj 1 $graph_dir $fold/val $fold/exp/mono_ali/decode
# 		#steps/train_mono.sh --config $helper_path/train_mono.config --nj 12 --cmd utils/run.pl $fold/train $fold/lang $fold/exp/$mono
# 		#steps/align_si.sh --nj 12 --cmd utils/run.pl $fold/train $fold/lang $fold/exp/$mono $fold/exp/$mono_ali
# 	done
# fi


if [ $ASR_stage -le 2 ]; then
	echo "Average best WER"
	python $helper_path/compute_wer.py $SS_kaldi mono_ali_clinician $SS_kaldi/set0/best_wer_all_folds_clinicians_unigram
	#python $helper_path/compute_wer.py $SS_kaldi mono_ali $SS_kaldi/set0/best_wer_all_folds
fi

# # #mkgraph
# if [ $ASR_stage -le 2 ]; then
# 	for fold in $SS_kaldi/set*; do
# 		echo "Mkgraph Fisher-English"
# 		utils/mkgraph.sh $fisher_root/data/lang_pp_test $fisher_root/exp/chain/blstm_7b $fold/exp/fisher/graph
# 	done
# fi

# if [ $ASR_stage -le 3 ]; then
# 	for fold in $SS_kaldi/set*; do
# 		cp $fisher_root/exp/chain/blstm_7b/final.mdl $fold/exp/fisher
# 		echo "Decode Fisher-English"
# 		steps/decode.sh --nj 1 $fold/exp/fisher/graph $fold/val $fold/exp/fisher/decode
# 		exit
# 		local/score.sh \
# 			$fold/val \
# 			$fold/exp/fisher/graph \
# 			$fold/exp/fisher/decode \
# 			|| exit 1;
# 	done
# fi

# if [ $ASR_stage -le 4 ]; then
# 	echo "Ger WER for val spkr over all folds"
# 	python $helper_path/get_WER_HC_HD.py $GF_kaldi $mono_ali
# fi

# if [ $ASR_stage -le 5 ]; then
# 	for sname in $data_type; do
# 		for fold in $GF_kaldi/set*; done
# 			echo "Get Phone/Word Alignment"
# 			ali_dir=$fold/exp/$mono_ali

# 			#Get ASR transcription and phone-level alignments
# 			lattice-best-path --acoustic-scale=0.1 \
# 			"ark:gunzip -c $fold/exp/$mono_ali/decode_$sname/lat.1.gz |" \
# 			"ark,t:|utils/int2sym.pl -f 2- $fold/lang/words.txt > $ali_dir/lattice_best_text.txt" \
# 			ark:- | \
# 			ali-to-phones --write-lengths $fold/exp/$mono_ali/final.mdl ark:- ark,t:$ali_dir/lattice_best_ali.txt


# 			ali="$ali_dir/lattice_best_ali.txt"
# 			text="$ali_dir/lattice_best_text.txt"
# 			phone2id="$fold/lang/phones.txt"
# 			lexicon="$fold/dict/lexicon.txt"
# 			python $CHAI_SHARE_PATH/Bins/kaldi/phone2word_ali.py $ali $text $phone2id $lexicon --sil-phones 1 2 3 4 5 > $ali_dir/word_ali_$sname.txt
# 		done
# 	done
# fi

