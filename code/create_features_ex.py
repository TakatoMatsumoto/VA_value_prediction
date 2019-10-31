# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import numpy as np
import librosa
from librosa import feature
import os
from glob import glob
import csv

import warnings
warnings.simplefilter('ignore')

def get_feature_vector(y, sr, S, song_id):
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
    melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    rmse = librosa.feature.rms(y=y)
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    poly_features = librosa.feature.poly_features(S=S, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    harmonic = librosa.effects.harmonic(y)
    percussive = librosa.effects.percussive(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    mfcc_delta = librosa.feature.delta(mfcc)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    frames_to_time = librosa.frames_to_time(onset_frames[:20], sr=sr)
    tempogram = librosa.feature.rhythm.tempogram(y=y, sr=sr)
    fourier_tempogram = librosa.feature.rhythm.fourier_tempogram(y=y, sr=sr)

    feature_set = {}
    feature_set['song_id'] = song_id
    feature_set['tempo'] = tempo  # tempo
    feature_set['average_beats'] = np.average(beats)
    feature_set['chroma_stft_mean'] = np.mean(chroma_stft)  # chroma stft
    feature_set['chroma_stft_std'] = np.std(chroma_stft)
    feature_set['chroma_cq_mean'] = np.mean(chroma_cq)  # chroma cq
    feature_set['chroma_cq_std'] = np.std(chroma_cq)
    feature_set['chroma_cens_mean'] = np.mean(chroma_cens)  # chroma cens
    feature_set['chroma_cens_std'] = np.std(chroma_cens)
    feature_set['melspectrogram_mean'] = np.mean(melspectrogram)  # melspectrogram
    feature_set['melspectrogram_std'] = np.std(melspectrogram)
    feature_set['rms_mean'] = np.mean(rmse)
    feature_set['rms_std'] = np.std(rmse)
    feature_set['mfcc_mean'] = np.mean(mfcc)  # mfcc
    feature_set['mfcc_std'] = np.std(mfcc)
    feature_set['mfcc_delta_mean'] = np.mean(mfcc_delta)  # mfcc delta
    feature_set['mfcc_delta_std'] = np.std(mfcc_delta)
    feature_set['cent_mean'] = np.mean(cent)  # cent
    feature_set['cent_std'] = np.std(cent)
    feature_set['spec_bw_mean'] = np.mean(spec_bw)  # spectral bandwidth
    feature_set['spec_bw_std'] = np.std(spec_bw)
    feature_set['contrast_mean'] = np.mean(contrast)  # contrast
    feature_set['contrast_std'] = np.std(contrast)
    feature_set['rolloff_mean'] = np.mean(rolloff)  # rolloff
    feature_set['rolloff_std'] = np.std(rolloff)
    feature_set['poly_mean'] = np.mean(poly_features)  # poly features
    feature_set['poly_std'] = np.std(poly_features)
    feature_set['tonnetz_mean'] = np.mean(tonnetz)  # tonnetz
    feature_set['tonnetz_std'] = np.std(tonnetz)
    feature_set['zcr_mean'] = np.mean(zcr)  # zero crossing rate
    feature_set['zcr_std'] = np.std(zcr)
    feature_set['harm_mean'] = np.mean(harmonic)  # harmonic
    feature_set['harm_std'] = np.std(harmonic)
    feature_set['perc_mean'] = np.mean(percussive)  # percussive
    feature_set['perc_std'] = np.std(percussive)
    feature_set['frame_mean'] = np.mean(frames_to_time)  # frames
    feature_set['frame_std'] = np.std(frames_to_time)
    feature_set['tempogram_mean'] = np.mean(tempogram)
    feature_set['tempogram_std'] = np.std(tempogram)
    feature_set['fourier_tempogram_mean_real'] = np.mean(fourier_tempogram).real
    feature_set['fourier_tempogram_std_real'] = np.std(fourier_tempogram).real
    return pd.Series(feature_set)

if __name__ == '__main__':
    audio_path = os.path.join("../input","audio_/")
    audio_files = glob(audio_path + '*.mp3')

    features = {}
    for file in audio_files:
        song_id = Path(file).stem
        print(song_id)
        y , sr = librosa.load(file, sr=None)
        S = np.abs(librosa.stft(y))
        feature_vector = get_feature_vector(y, sr, S, song_id)
        features = pd.concat([features, feature_vector])

    print(features)
