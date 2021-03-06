# Extract phonation features
import os
import sys
import webrtcvad
import contextlib
import wave
import collections
import pandas as pd
import subprocess
sys.path.insert(0, sys.path[0]+'/VAD-python')
from vad import VoiceActivityDetector
import sox

early_late_balanced = {'44574': 3, '95465': 0, '13691': 0, '82697': 0, '35063': 3, '11739': 0, '76373': 3,
'38717': 2, '68117': 0, '00359': 2, '81392': 3, '07920': 0, '95407': 3, '56896': 3, '24371': 2, '78597': 2,
'18261': 2, '53870': 2, '61496': 0, '88947': 0, '47647': 0, '32762': 2, '25066': 0, '16486': 0, '18771': 0,
'45758': 0, '50377': 2, '26753': 0, '29735': 0, '73828': 2, '47939': 0, '80292': 3, '55029': 2, '58812': 0,
'44209': 0, '42080': 2, '05068': 0, '33752': 2}

def apply_VAD(phonation_files, new_wav_path):
    # speech_HD = {0:[0,0], 2:[0,0], 3:[0,0]}
    # for wav in phonation_files:
    #     optimal_thresh = 0.5
    #     v = VoiceActivityDetector(wav)
    #     v.speech_energy_threshold = optimal_thresh
    #     frames = v.detect_speech()
    #     speech_labels = v.convert_windows_to_readible_labels(frames)
    #     # print('spkr wav', wav.split('/')[-1])
    #     # print(speech_labels)
    #     # exit()
    #     spkr_id = wav.split('/')[-1].split('_')[0]
    #     if len(speech_labels) > 1: #if more than 1 split
    #         speech_HD[early_late_balanced[spkr_id]][1]+=1
    #     else:
    #         speech_HD[early_late_balanced[spkr_id]][0]+=1
    # print(speech_HD)
    # exit()

    for thresh_sweep in [0.25, 0.3, 0.35]: #0.5 too high, no speech detected for some spkrs
        for wav in phonation_files:
            optimal_thresh = thresh_sweep
            v = VoiceActivityDetector(wav)
            v.speech_energy_threshold = optimal_thresh
            frames = v.detect_speech()
            speech_labels = v.convert_windows_to_readible_labels(frames)

            #output new wav
            old_basename = wav.split('/')[-1]
            new_wav_path_dir = os.path.join(new_wav_path, str(thresh_sweep))
            if not os.path.exists(new_wav_path_dir):
                os.makedirs(new_wav_path_dir)
            out_wav = os.path.join(new_wav_path_dir, old_basename)

            # print(frames)
            # for i in frames:
            #     if i[1] > 0:
            #         print(i)
            # print('wav', wav)
            # print(speech_labels)
            # print(thresh_sweep)
            if '38717' in wav:
                print(speech_labels)
                exit()
            first_vad_instance = speech_labels[0]
            beg_t = first_vad_instance['speech_begin']
            end_t = first_vad_instance['speech_end']

            transformer = sox.Transformer()
            transformer.trim(float(beg_t),float(end_t))
            transformer.convert(samplerate=32000, n_channels=1)
            transformer.build(wav, out_wav)

def extract_phonation_features(conf_file, input_file, output_file):
    smile_root = '/z/public/openSMILE/SMILExtract'
    
    print('cmd', " ".join([smile_root,
                    '-C', conf_file,
                    '-I', input_file,
                    '-O', output_file]))


    p = subprocess.Popen([smile_root,
                    '-C', conf_file,
                    '-I', input_file,
                    '-O', output_file], stdout=subprocess.PIPE).communicate()[0]

def comb_feats(feats_base, out_comb_csv, v_seg):
    #go through, check for csv
    csvs = [f for f in os.listdir(feats_base) if f.endswith(".csv") and not f.startswith("comb") and v_seg in f]
    # print(csvs)
    # exit()

    df = pd.DataFrame()
    for csv in csvs:
        #read into df
        full_csv_path = os.path.join(feats_base, csv)
        spkr_id = csv.split('_')[0]
        # print(spkr_id)
        temp_df = pd.read_csv(full_csv_path)
        temp_df['id'] = pd.Series(spkr_id)

        # print('null', temp_df.isnull().any(axis=1))
        # print('null', temp_df)
        # exit()
        df = df.append(temp_df)

    #set index as 'id'
    df = df.set_index('id')
    
    # print('index', df['id'])
    print('null', df.isnull().any(axis=1))
    exit()
    #compute unvoiced segments
    df['NUV'] = df['F0_sma_duration'] - df['F0_sma_nnz']
    df['UV-V-portion'] = df['NUV'] / df['F0_sma_duration']


    #drop uplevel downlevel
    drop_leveltime = [i for i in df.columns.values if 'downleveltime' in i or 'upleveltime' in i]
    df = df.drop(drop_leveltime, axis=1)

    #drop nnz and duration for everything by F0
    dur_key = 'F0_sma_duration'
    drop_duration = [i for i in df.columns.values if 'duration' in i and i != dur_key]
    df = df.drop(drop_duration, axis=1)
    nnz_key = 'F0_sma_nnz'
    drop_nnz = [i for i in df.columns.values if 'nnz' in i and i != nnz_key]
    df = df.drop(drop_nnz, axis=1)

    df = df.dropna(axis=1) #drop nan values

    #write dataframe to directory
    df.to_csv(out_comb_csv)


def main():
    phonation_data = sys.argv[1]
    feats_path = sys.argv[2]
    openSMILE_conf = sys.argv[3]

    segments_file = os.path.join(phonation_data, 'segments')
    seg_wavs = os.path.join(phonation_data, 'seg_wavs')
    phonation_files = [os.path.join(seg_wavs, i) for i in os.listdir(seg_wavs) if i.endswith('.wav')]

    # Apply VAD
    vad_seg_wavs = os.path.join(phonation_data, 'vad_seg_wavs')
    if not os.path.exists(os.path.join(phonation_data, 'vad_seg_wavs')):
        os.makedirs(os.path.join(phonation_data, 'vad_seg_wavs'))

    apply_VAD(phonation_files, vad_seg_wavs)
    exit()

    ### extract features
    # sweep voiceProb cutoff
    for cutoff in [25,50,75]:
        phonation_feats_out = os.path.join(feats_path, str(cutoff)+'_phonation')
        if not os.path.exists(phonation_feats_out):
            os.makedirs(phonation_feats_out)

        openSMILE_conf_new = openSMILE_conf.split('.')[0] +'_'+ str(cutoff)+'.conf'

        #sweep vad segs
        for v_seg in ['0.25', '0.3', '0.35']:
            vad_wavs = os.path.join(phonation_data, 'vad_seg_wavs', v_seg)
            phonation_vad_files = [os.path.join(vad_wavs, i) for i in os.listdir(vad_wavs) if i.endswith('.wav')]

            for wav in phonation_vad_files:
                spkr_id = wav.split('/')[-1].split('_')[0]
                out_file = os.path.join(phonation_feats_out, spkr_id+'_'+v_seg+'.csv')
                extract_phonation_features(openSMILE_conf_new, wav, out_file)

            #Combine features
            comb_file = os.path.join(phonation_feats_out,'comb_feats_'+str(cutoff)+'_vad_'+str(v_seg)+'.csv')
            comb_feats(phonation_feats_out, comb_file, v_seg)
            # print('final', comb_file)
            exit()

if __name__ == '__main__':
    main()