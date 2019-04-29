#Important for calling kaldi scripts. Push to working directory
kd_root='/z/public/kaldi/egs/wsj/s5'
pushd $kd_root

#For build_lm.sh
. $kd_root/../../../tools/env.sh

root_='/z/mkperez/SpontSpeech-HD'
#scripts
scripts_path=$root_'/Scripts'
helper_path=$root_'/Scripts/kaldi/helper'

#Use data from GF Passage
raw_GF='/z/mkperez/Speech-Data-Raw/HD-GF'
GF_kaldi=$root_'/Data/GF-Passage'

raw_SS='/z/mkperez/Speech-Data-Raw/HD-SS'
SS_kaldi=$root_'/Data/Spont-Speech'

#End-to-End
EE_kaldi=$root_'/Data/E2E'