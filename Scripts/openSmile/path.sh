#Important for calling kaldi scripts. Push to working directory
kd_root='/z/public/kaldi/egs/wsj/s5'
pushd $kd_root

#For build_lm.sh
. $kd_root/../../../tools/env.sh

root_='/z/mkperez/MPT_prosodic'
#scripts
scripts_path=$root_'/Scripts'
helper_path=$root_'/Scripts/openSmile/helper'

#Data
raw_SS_wav='/z/mkperez/Speech-Data-Raw/HD-SS'
SS_segmented_wav='/z/mkperez/Speech-Data-Raw/HD-SS/segmented'
SS_Data=$root_'/Data'
SS_Phonation=$SS_Data'/Spont-Speech_phonation_manifest'

#Features
SS_feats=$root_'/Features'

#OpenSMILE
OpenSMILE_dir='/z/public/openSMILE'
openSMILE_cfg=$helper_path'/openSMILE_prosodic.conf'

#results
SS_results=$root_'/Results'