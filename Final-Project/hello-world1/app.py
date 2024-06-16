"""
A sample Hello World server.
"""
from flask import Flask, render_template, request, jsonify
import threading
import time
import numpy as np
import scipy.fftpack
import sounddevice as sd
import copy
import os
from flask import Flask, render_template

# pylint: disable=C0103
app = Flask(__name__)

# General settings contants for audio processing
SAMPLE_FREQ = 48000 # sample frequency in Hz
WINDOW_SIZE = 48000 # window size of the DFT in samples
WINDOW_STEP = 12000 # step size of window
NUM_HPS = 5 # max number of harmonic product spectrums
POWER_THRESH = 1e-6 # tuning is activated if the signal power exceeds this threshold
CONCERT_PITCH = 440 # defining a1
WHITE_NOISE_THRESH = 0.2 # everything under WHITE_NOISE_THRESH*avg_energy_per_freq is cut off

WINDOW_T_LEN = WINDOW_SIZE / SAMPLE_FREQ # length of the window in seconds
SAMPLE_T_LENGTH = 1 / SAMPLE_FREQ # length between two samples in seconds
DELTA_FREQ = SAMPLE_FREQ / WINDOW_SIZE # frequency step width of the interpolated DFT
OCTAVE_BANDS = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]

ALL_NOTES = ["A","A#","B","C","C#","D","D#","E","F","F#","G","G#"]
NOTE_TO_TAB = {
    'E2': (6, 0), 'F2': (6, 1), 'F#2': (6, 2), 'G2': (6, 3), 'G#2': (6, 4), 'A2': (6, 5), 'A#2': (6, 6), 'B2': (6, 7), 'C3': (6, 8), 'C#3': (6, 9), 'D3': (6, 10), 'D#3': (6, 11), 'E3': (6, 12),
    'A2': (5, 0), 'A#2': (5, 1), 'B2': (5, 2), 'C3': (5, 3), 'C#3': (5, 4), 'D3': (5, 5), 'D#3': (5, 6), 'E3': (5, 7), 'F3': (5, 8), 'F#3': (5, 9), 'G3': (5, 10), 'G#3': (5, 11), 'A3': (5, 12),
    'D3': (4, 0), 'D#3': (4, 1), 'E3': (4, 2), 'F3': (4, 3), 'F#3': (4, 4), 'G3': (4, 5), 'G#3': (4, 6), 'A3': (4, 7), 'A#3': (4, 8), 'B3': (4, 9), 'C4': (4, 10), 'C#4': (4, 11), 'D4': (4, 12),
    'G3': (3, 0), 'G#3': (3, 1), 'A3': (3, 2), 'A#3': (3, 3), 'B3': (3, 4), 'C4': (3, 5), 'C#4': (3, 6), 'D4': (3, 7), 'D#4': (3, 8), 'E4': (3, 9), 'F4': (3, 10), 'F#4': (3, 11), 'G4': (3, 12),
    'B3': (2, 0), 'C4': (2, 1), 'C#4': (2, 2), 'D4': (2, 3), 'D#4': (2, 4), 'E4': (2, 5), 'F4': (2, 6), 'F#4': (2, 7), 'G4': (2, 8), 'G#4': (2, 9), 'A4': (2, 10), 'A#4': (2, 11), 'B4': (2, 12),
    'E4': (1, 0), 'F4': (1, 1), 'F#4': (1, 2), 'G4': (1, 3), 'G#4': (1, 4), 'A4': (1, 5), 'A#4': (1, 6), 'B4': (1, 7), 'C5': (1, 8), 'C#5': (1, 9), 'D5': (1, 10), 'D#5': (1, 11), 'E5': (1, 12),
}

def find_closest_note(pitch):
    i = int(np.round(np.log2(pitch/CONCERT_PITCH)*12))
    closest_note = ALL_NOTES[i%12] + str(4 + (i + 9) // 12)
    closest_pitch = CONCERT_PITCH*2**(i/12)
    return closest_note, closest_pitch

HANN_WINDOW = np.hanning(WINDOW_SIZE)

# Global variable to store detected notes and tablature
detected_notes = []
tuning_active = False
tuner_thread = None

def note_to_tab(note):
    return NOTE_TO_TAB.get(note, None)

def callback(indata, frames, time, status):
    if not hasattr(callback, "window_samples"):
        callback.window_samples = [0 for _ in range(WINDOW_SIZE)]
    if not hasattr(callback, "noteBuffer"):
        callback.noteBuffer = ["1","2"]

    if status:
        print(status)
        return
    if any(indata):
        callback.window_samples = np.concatenate((callback.window_samples, indata[:, 0])) # append new samples
        callback.window_samples = callback.window_samples[len(indata[:, 0]):] # remove old samples

        signal_power = (np.linalg.norm(callback.window_samples, ord=2)**2) / len(callback.window_samples)
        if signal_power < POWER_THRESH:
            return

        hann_samples = callback.window_samples * HANN_WINDOW
        magnitude_spec = abs(scipy.fftpack.fft(hann_samples)[:len(hann_samples)//2])

        for i in range(int(62/DELTA_FREQ)):
            magnitude_spec[i] = 0

        for j in range(len(OCTAVE_BANDS)-1):
            ind_start = int(OCTAVE_BANDS[j]/DELTA_FREQ)
            ind_end = int(OCTAVE_BANDS[j+1]/DELTA_FREQ)
            ind_end = ind_end if len(magnitude_spec) > ind_end else len(magnitude_spec)
            avg_energy_per_freq = (np.linalg.norm(magnitude_spec[ind_start:ind_end], ord=2)**2) / (ind_end-ind_start)
            avg_energy_per_freq = avg_energy_per_freq**0.5
            for i in range(ind_start, ind_end):
                magnitude_spec[i] = magnitude_spec[i] if magnitude_spec[i] > WHITE_NOISE_THRESH*avg_energy_per_freq else 0

        mag_spec_ipol = np.interp(np.arange(0, len(magnitude_spec), 1/NUM_HPS), np.arange(0, len(magnitude_spec)), magnitude_spec)
        mag_spec_ipol = mag_spec_ipol / np.linalg.norm(mag_spec_ipol, ord=2)

        hps_spec = copy.deepcopy(mag_spec_ipol)

        for i in range(NUM_HPS):
            tmp_hps_spec = np.multiply(hps_spec[:int(np.ceil(len(mag_spec_ipol)/(i+1)))], mag_spec_ipol[::(i+1)])
            if not any(tmp_hps_spec):
                break
            hps_spec = tmp_hps_spec

        max_ind = np.argmax(hps_spec)
        max_freq = max_ind * (SAMPLE_FREQ/WINDOW_SIZE) / NUM_HPS

        closest_note, closest_pitch = find_closest_note(max_freq)
        max_freq = round(max_freq, 1)
        closest_pitch = round(closest_pitch, 1)

        callback.noteBuffer.insert(0, closest_note)
        callback.noteBuffer.pop()

        if callback.noteBuffer.count(callback.noteBuffer[0]) == len(callback.noteBuffer):
            tab = note_to_tab(closest_note)
            if tab:
                detected_notes.append((closest_note, tab))
        else:
            return

    else:
        print('no input')

def tuner():
    global tuning_active
    try:
        with sd.InputStream(channels=1, callback=callback, blocksize=WINDOW_STEP, samplerate=SAMPLE_FREQ):
            while tuning_active:
                time.sleep(0.5)
    except Exception as e:
        print(str(e))

@app.route('/')
def index():
    """Return a friendly HTTP greeting."""
    # message = "This the Hello World Program!"

    """Get Cloud Run environment variables."""
    service = os.environ.get('K_SERVICE', 'Unknown service')
    revision = os.environ.get('K_REVISION', 'Unknown revision')

    return render_template('index.html',
        Service=service,
        Revision=revision)

@app.route('/start', methods=['POST'])
def start_tuner():
    global tuning_active, tuner_thread, detected_notes
    detected_notes = []
    if not tuning_active:
        tuning_active = True
        tuner_thread = threading.Thread(target=tuner)
        tuner_thread.start()
    return jsonify({"status": "Recording started"})

@app.route('/stop', methods=['POST'])
def stop_tuner():
    global tuning_active
    tuning_active = False
    if tuner_thread:
        tuner_thread.join()
    notes = [{'note': note, 'string': tab[0], 'fret': tab[1]} for note, tab in detected_notes]
    return jsonify({"status": "Tuner stopped", "notes": notes})


if __name__ == '__main__':
    server_port = os.environ.get('PORT', '8080')
    app.run(debug=False, port=server_port, host='0.0.0.0')
