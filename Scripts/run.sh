#Matthew Perez
. kaldi/path.sh
stage=1
data_type="train val"
mono_type="mono_2000_ali"
decode_nj=1

#Feature Extraction
if [ $stage -le 1 ]; then
	for data in $GF_kaldi/set*; do
		echo $data
		for sname in $data_type; do
			echo "Extracting GOP Features"
			mkdir -p $data/$sname/acoustic_lexical_features/gop
			
			#need to execute in the egs directory
			pushd $helper_path/kaldi-gop/egs/gop-compute
			local/compute-gmm-gop.sh $data/$sname $data/lang $data/exp/mono-2000_ali $data/$sname/acoustic_lexical_features/gop
			popd


			echo "Extracting Lexical Features"
			#Create Lexical features and combine with GOP features
			python $helper_path/feat_extract.py $data/exp/mono-2000_ali $sname $data/$sname/acoustic_lexical_features/gop/gop.1 $data/$sname/acoustic_lexical_features
			#exit
		done
	done
fi

#Model Training