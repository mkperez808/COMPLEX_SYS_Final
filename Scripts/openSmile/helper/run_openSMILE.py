import subprocess
import sys
import os

# PROJECT_ROOT = '/home/aldeneh/priori_turns/extract_features'
# INPUT_FILE = PROJECT_ROOT + '/wav_samples/OSR_us_000_0010_8k.wav'
# OUTPUT_FILE = PROJECT_ROOT + '/output/out.csv'
# CONF_FILE = PROJECT_ROOT + '/config/common_feats.conf'


def run_opensmile(smile_root, level, conf_file, input_file, output_file):
    # one_line = " ".join([smile_root,
    #                     '-C', conf_file,
    #                     '-l', level,
    #                     '-I', input_file,
    #                     '-O', output_file])
    # message = subprocess.check_output(one_line, shell=True, stderr=subprocess.STDOUT)
    # print(message)
    # print('cmd', " ".join([smile_root,
    #                      '-C', conf_file,
    #                      '-I', input_file,
    #                      '-O', output_file]))

    print('cmd', " ".join([smile_root,
                    '-C', conf_file,
                    '-l', level,
                    '-I', input_file,
                    '-O', output_file]))


    # p = subprocess.Popen([smile_root,
    #                 '-C', conf_file,
    #                 '-l', level,
    #                 '-I', input_file,
    #                 '-O', output_file], stdout=subprocess.PIPE).communicate()[0]


    p = subprocess.call([smile_root,
                    '--help',
                    '-C', conf_file,
                    '-l', level,
                    '-I', input_file,
                    '-O', output_file], stdout=subprocess.PIPE)


    #stdout, stderr = p.communicate()
    #log_subprocess_output(StringIO(stderr))
    #print('stderr', stderr)
    #print('stdout', stdout)

if __name__ == '__main__':
    conf_file = sys.argv[1]
    input_wav = sys.argv[2]
    utt_id = sys.argv[3].split('.')[0]
    feat_base = os.path.join(sys.argv[4], "openSMILE")
    if not os.path.exists(feat_base):
        os.makedirs(feat_base)

    out_csv = os.path.join(feat_base, utt_id+'.csv')
    # print(input_wav)
    # exit()

    SMILE_ROOT = '/z/public/openSMILE/SMILExtract'

    demo_wav = '/z/public/openSMILE/example-audio/opensmile.wav'
    demo_conf = '/z/public/openSMILE/config/demo/demo1_energy.conf'
    run_opensmile(SMILE_ROOT, '2', demo_conf, demo_wav, out_csv)
