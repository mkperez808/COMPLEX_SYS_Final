# Extract pause features using VAD
import os
import sys
import webrtcvad
import contextlib
import wave
import collections
import pandas as pd
sys.path.insert(0, sys.path[0]+'/VAD-python')
from vad import VoiceActivityDetector

early_late_balanced = {'44574': 3, '95465': 0, '13691': 0, '82697': 0, '35063': 3, '11739': 0, '76373': 3,
'38717': 2, '68117': 0, '00359': 2, '81392': 3, '07920': 0, '95407': 3, '56896': 3, '24371': 2, '78597': 2,
'18261': 2, '53870': 2, '61496': 0, '88947': 0, '47647': 0, '32762': 2, '25066': 0, '16486': 0, '18771': 0,
'45758': 0, '50377': 2, '26753': 0, '29735': 0, '73828': 2, '47939': 0, '80292': 3, '55029': 2, '58812': 0,
'44209': 0, '42080': 2, '05068': 0, '33752': 2}

late_balanced = {'47647': 0, '56896': 3, '81392': 3, '26753': 0, '44574': 3, '35063': 3, '18771': 0,
'76373': 3, '45758': 0, '47939': 0, '80292': 3, '58812': 0, '68117': 0, '95407': 3}

def read_wave(path):
    """Reads a .wav file.
    Takes the path, and returns (PCM audio data, sample rate).
    """
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate

class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration

def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.
    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.
    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n

def vad_collector(sample_rate, frame_duration_ms,
                  padding_duration_ms, vad, frames):
    """Filters out non-voiced audio frames.
    Given a webrtcvad.Vad and a source of audio frames, yields only
    the voiced audio.
    Uses a padded, sliding window algorithm over the audio frames.
    When more than 90% of the frames in the window are voiced (as
    reported by the VAD), the collector triggers and begins yielding
    audio frames. Then the collector waits until 90% of the frames in
    the window are unvoiced to detrigger.
    The window is padded at the front and back to provide a small
    amount of silence or the beginnings/endings of speech around the
    voiced frames.
    Arguments:
    sample_rate - The audio sample rate, in Hz.
    frame_duration_ms - The frame duration in milliseconds.
    padding_duration_ms - The amount to pad the window, in milliseconds.
    vad - An instance of webrtcvad.Vad.
    frames - a source of audio frames (sequence or generator).
    Returns: A generator that yields PCM audio data.
    """
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
    # NOTTRIGGERED state.
    triggered = False

    voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        sys.stdout.write('1' if is_speech else '0')
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            # If we're NOTTRIGGERED and more than 90% of the frames in
            # the ring buffer are voiced frames, then enter the
            # TRIGGERED state.
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                sys.stdout.write('+(%s)' % (ring_buffer[0][0].timestamp,))
                # We want to yield all the audio we see from now until
                # we are NOTTRIGGERED, but we have to start with the
                # audio that's already in the ring buffer.
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            # We're in the TRIGGERED state, so collect the audio data
            # and add it to the ring buffer.
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            # If more than 90% of the frames in the ring buffer are
            # unvoiced, then enter NOTTRIGGERED and yield whatever
            # audio we've collected.
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
                triggered = False
                yield b''.join([f.bytes for f in voiced_frames])
                ring_buffer.clear()
                voiced_frames = []
    if triggered:
        sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
    sys.stdout.write('\n')
    # If we have any leftover voiced audio when we run out of input,
    # yield it.
    if voiced_frames:
        yield b''.join([f.bytes for f in voiced_frames])

def compute_pause_feats(frames):
    # frames: 1=speech, 0 = non-speech
    # num_pauses / time
    # pause frames / time
    result = {}

    num_frames = len(frames)
    pause_feats = frames[:,1]
    num_vcd = len([1 for i in pause_feats if i>0])
    num_pause = len([0 for i in pause_feats if i<1])

    #Total utterance pause features
    result['pause_portion'] = float(num_pause) / float(num_frames)
    result['vcd_portion'] = float(num_vcd) / float(num_frames)

    # number of fluctuations => gaps in speech (pronunciation/cognition)
    switches = len([1 for idx, p in enumerate(pause_feats[:-1]) if p!=pause_feats[idx+1]])
    result['pause-vcd-switches'] = switches

    # print(pause_feats, num_vcd, num_pause, switches)
    return result

# def sweep_vad_threshold(default, path):
#     for i in range(40): #max 40 iterations of tuning
#         v = VoiceActivityDetector(path)
#         v.speech_energy_threshold = default
#         frames = v.detect_speech()

#         features = frames[:1]
#         tot_frames = len(frames[:1])
#         if len([1 for x in features if x>0]) < 0.6 * 


def main():
    wav_path = sys.argv[1]
    feats_path = sys.argv[2]
    label_type = sys.argv[3]
    seg_wavs = os.path.join(wav_path, "segmented")

    # print(sorted(os.listdir(seg_wavs)))
    # exit()
    df = pd.DataFrame()
    for wav in sorted(os.listdir(seg_wavs)):

        #optimal_thresh = sweep_vad_threshold(default=0.3, path=os.path.join(seg_wavs, wav))
        optimal_thresh = 0.3
        v = VoiceActivityDetector(os.path.join(seg_wavs, wav))
        v.speech_energy_threshold = optimal_thresh
        frames = v.detect_speech()
        # print(wav, frames)

        utt_id = wav.split(".")[0]
        result = compute_pause_feats(frames)
        result['id'] = utt_id
        df = df.append(result, ignore_index=True)

    df = df.set_index('id')
    df.to_csv(os.path.join(feats_path,label_type+'_pause_feats.csv'))
    print(" ".join(late_balanced.keys()))
    print(" ".join(early_late_balanced.keys()))


        # exit()

        # vad = webrtcvad.Vad(2)
        # audio, sample_rate = read_wave(os.path.join(seg_wavs, wav))
        # frames = frame_generator(30, audio, sample_rate)
        # frames = list(frames)
        # segments = vad_collector(sample_rate, 30, 300, vad, frames)
        # for i, segment in enumerate(segments):
        #    # path = 'chunk-%002d.wav' % (i,)
        #     print(' chunk%s %s' % (i,segment))
        #     #write_wave(path, segment, sample_rate)





if __name__ == '__main__':
    main()