#Matthew Perez
. ./path.sh
stage=1
ASR_stage=5
data_type="train val"


########  KALDI PRE-SETUP  ##################

if [ $stage -le 1 ]; then
	pushd $helper_path
	python data_preprocessing.py $raw_GF $GF_kaldi "true"
	popd
fi

if [ $stage -le 2 ]; then
	echo "Copying Dictionaries"
	pushd $helper_path
	./copy_dict.sh
	popd
fi

if [ $stage -le 3 ]; then
	echo "Spk2Utt prep"
	for sname in $data_type; do
		for data in $GF_kaldi/set*; do
			utils/utt2spk_to_spk2utt.pl $data/$sname/utt2spk > $data/$sname/spk2utt
		done
	done
fi

if [ $stage -le 4 ]; then
	echo "Validate data directories"
	for sname in $data_type; do
		for data in $GF_kaldi/set*; do
			utils/validate_data_dir.sh $data/$sname
			utils/fix_data_dir.sh $data/$sname
			#exit
		done
	done
fi
exit
if [ $stage -le 5 ]; then
	echo "Feature creation (MFCC)"
	for sname in $data_type; do
		for data in $GF_kaldi/set*; do
			steps/make_mfcc.sh --nj 4 --cmd utils/run.pl $data/$sname $data/$sname/mfcc/log $data/$sname/mfcc
			steps/compute_cmvn_stats.sh $data/$sname $data/$sname/mfcc/log $data/$sname/mfcc
		done
	done
fi

if [ $stage -le 6 ]; then
	echo "Generate bigram text"
	for data in $GF_kaldi/set*; do
		pushd $scripts_path/kaldi/helper
		python gen-bigram-text.py $data/train/text $data/train/LM_text
		popd
		#exit
	done
fi

if [ $stage -le 7 ]; then
	echo "Prepare /lang directory"
	for data in $GF_kaldi/set*; do
		utils/prepare_lang.sh $data/dict '<UNK>' $data/local $data/lang
		#exit
	done
fi

if [ $stage -le 8 ]; then
	echo "Create Langauge Model (G.fst)"
	for data in $GF_kaldi/set*; do
		$CHAI_SHARE_PATH/Bins/kaldi/build_lm.sh $data/train/LM_text $data/lang
		#exit
	done
fi

################################################

############  ASR Training   ############
mono='mono-2000'
mono_ali=$mono'_ali'
if [ $ASR_stage -le 1 ]; then
	for fold in $GF_kaldi/set*; do
		echo "Monophone training"
		steps/train_mono.sh --config $helper_path/train_mono.config --nj 12 --cmd utils/run.pl $fold/train $fold/lang $fold/exp/$mono
		steps/align_si.sh --nj 12 --cmd utils/run.pl $fold/train $fold/lang $fold/exp/$mono $fold/exp/$mono_ali
	done
fi

#mkgraph
if [ $ASR_stage -le 2 ]; then
	for fold in $GF_kaldi/set*; do
		echo "Mkgraph"
		utils/mkgraph.sh $fold/lang $fold/exp/$mono_ali $fold/exp/$mono_ali/graph
	done
fi

if [ $ASR_stage -le 3 ]; then
	for sname in $data_type; do
		for fold in $GF_kaldi/set*; do
			echo "Decode Monophone"
			steps/decode.sh --nj 1 $fold/exp/$mono_ali/graph $fold/$sname $fold/exp/$mono_ali/decode_$sname
			local/score.sh \
				$fold/$sname \
				$fold/exp/$mono_ali/graph \
				$fold/exp/$mono_ali/decode_$sname \
				|| exit 1;
		done
	done
fi

if [ $ASR_stage -le 4 ]; then
	echo "Ger WER for val spkr over all folds"
	python $helper_path/get_WER_HC_HD.py $GF_kaldi $mono_ali
fi

if [ $ASR_stage -le 5 ]; then
	for sname in $data_type; do
		for fold in $GF_kaldi/set*; done
			echo "Get Phone/Word Alignment"
			ali_dir=$fold/exp/$mono_ali

			#Get ASR transcription and phone-level alignments
			lattice-best-path --acoustic-scale=0.1 \
			"ark:gunzip -c $fold/exp/$mono_ali/decode_$sname/lat.1.gz |" \
			"ark,t:|utils/int2sym.pl -f 2- $fold/lang/words.txt > $ali_dir/lattice_best_text.txt" \
			ark:- | \
			ali-to-phones --write-lengths $fold/exp/$mono_ali/final.mdl ark:- ark,t:$ali_dir/lattice_best_ali.txt


			ali="$ali_dir/lattice_best_ali.txt"
			text="$ali_dir/lattice_best_text.txt"
			phone2id="$fold/lang/phones.txt"
			lexicon="$fold/dict/lexicon.txt"
			python $CHAI_SHARE_PATH/Bins/kaldi/phone2word_ali.py $ali $text $phone2id $lexicon --sil-phones 1 2 3 4 5 > $ali_dir/word_ali_$sname.txt
		done
	done
fi