#%%
from scipy.io.wavfile import read
import numpy as np
import librosa
import librosa.display
import pysptk
from spectrum import *
from Daniell import *

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rc

from scipy import signal

mpl.rcParams['font.size'] = 18.0
mpl.rcParams['axes.titlesize'] = 18.0; mpl.rcParams['axes.labelsize'] = 18.0
mpl.rcParams["figure.facecolor"] = 'darkslategray'
plt.style.use('dark_background')
plt.rcParams['axes.facecolor'] = 'darkslategray'
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['xtick.labelsize'] = 'xx-large'
plt.rc('xtick', labelsize='x-large'); plt.rc('ytick', labelsize='x-large')
plt.rc('axes', labelsize='x-large', titlesize='x-large')

#%%
from argparse import ArgumentParser, RawTextHelpFormatter
parser = ArgumentParser(description="Audio PSD Analysis.", formatter_class=RawTextHelpFormatter)
parser.add_argument('-A', '--audio', metavar='Filename', type=str, nargs='?',
                    help='Filename of audio', 
                    required=False, default = 'fluteircam.wav')
args = parser.parse_args()
audio = args.audio; 

#%%
root = './assets/audio/'; file_name = audio
# Read the Audiofile
sr, y = read(root+file_name)
y = y.astype(float)
duration = len(y)/sr # # Duration of the audio in Seconds
time = np.arange(0,duration,1/sr)
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)  # Spectrogram in dB
start = 10000; N = 4000;

#%%

def simple_f0s(model, y, N, **kwargs):
    '''
    Estimate the pitch by the PSD Estimation model
    Arguments:
        model: PSD Estimation model
        y: original audio 
        N: the segments length
    '''
    f0s = []
    for i in np.arange(0, len(y)-N, N): 
        sample = y[i:i+N]
        p = model(sample, sampling=sr)
        psd = p.psd # contains the PSD values
        f_period = p.frequencies() # returns a list of frequencies
        f0 = f_period[np.argmax(psd)]
        f0s.append(f0)
    return np.array(f0s)


def f0s_spectro(spectro, sr, time_data, f0s_data, labels, ax_title, has_spectro=True):
    fig, ax = plt.subplots(nrows=1, figsize=(12,6))
    if has_spectro:
        img = librosa.display.specshow(spectro, x_axis='time', y_axis='log', ax=ax, sr=sr)
        fig.colorbar(img, ax=ax, format="%+2.f dB")
    ax.set(title=ax_title)
    for idx, f0s in enumerate(f0s_data):
        ax.step(time_data[idx], f0s, label=labels[idx], linewidth=3)
    ax.grid(axis='x', color='0.95')
    ax.legend(title='PSD Estimation Methods')
    ax.set_xlim(0,time_data[0][-1])
    plt.show()


#%% 1.1 Periodogram
f0s_p = simple_f0s(Periodogram, y, N)
time_p = linspace(0,duration,len(f0s_p))

#%% 1.2 Autocorrelation
#  frame size is the block size divided by sample frequency
f0_yin = librosa.yin(y=y, sr=sr, fmin=65, fmax=2093, frame_length=2048, trough_threshold=0.1)
f0_pyin, voiced_flag, voiced_probs = librosa.pyin(y, fmin=65, fmax=2093, frame_length=2048, 
                         n_thresholds=100, beta_parameters=(2, 18), boltzmann_parameter=2, 
                        resolution=0.1, max_transition_rate=35.92, switch_prob=0.01, no_trough_prob=0.0)
times_yin = librosa.times_like(f0_yin, sr=sr, hop_length=512, n_fft=None)

#%% Plot pYIN/YIN/Periodogram

f0s_data = np.array([f0_yin, f0_pyin, f0s_p])
time_data = np.array([times_yin, times_yin, time_p])
labels = ['f0_YIN', 'f0_pYIN', 'Periodogram']
ax_title = 'YIN & pYIN & Periodogram (f0s Estimation)'
f0s_spectro(D, sr, time_data, f0s_data, labels, ax_title, has_spectro=False)



#%% Non-Parametric

def plot_PSD(fs, Powers, title, labels, figsize=(10,6), x_label='frequency [Hz]', y_label='PSD [dB]'):
    plt.figure(figsize=figsize); plt.title(title)
    for idx, f in enumerate(fs):
        plt.plot(f, 10*log10(abs(Powers[idx])*2./(2.*pi)), label=labels[idx])
    plt.xlabel(x_label); plt.ylabel(y_label)
    plt.legend()
    
    
# n_overlap: If None, noverlap = nperseg // 2.
f0s_welch = []; f0s_daniell = []; f0s_mv = []
nperseg = 1024; P = 8; NFFT = 4096; order = 32
for i in np.arange(0, len(y)-N, N): 
    y_sample = y[i:i+N]
    f, psd_welch = signal.welch(y_sample, sr, window='hann', nperseg=nperseg, noverlap=None, 
                        nfft=NFFT, scaling='spectrum')
    f0 = f[np.argmax(psd_welch)]; f0s_welch.append(f0)
    psd_daniell, f_daniell = DaniellPeriodogram(y_sample, P=P, NFFT=NFFT, detrend='mean',
                       sampling=sr, scale_by_freq=False, window='hamming')
    f0 = f_daniell[np.argmax(psd_daniell)]; f0s_daniell.append(f0)
    if i == N*81:
        fs = np.array([f, f_daniell])
        Powers = np.array([psd_welch, psd_daniell])
        labels = np.array([f'Welch (nperseg {nperseg})', f'Daniell (nperseg {P})'])
        title = 'Non-Parametric PSD'
        plot_PSD(fs, Powers, title, labels)

f0s_spectro(D, sr, np.array([time_p, time_p]), np.array([f0s_welch, f0s_daniell]), 
            ['Welch', 'Daniell'], ax_title='Non-Parametric f0s', has_spectro=False)



#%% Yule-Walker & Burg

# yule-walker
order= 32; NFFT = 4096
f0s_yw = [] ; f0s_burg = []
for i in np.arange(0, len(y)-N, N): 
    y_sample = y[i:i+N]    
    yw_model = pyule(y_sample, order = order, NFFT=NFFT, sampling=sr, scale_by_freq=False)
    f0 = yw_model.frequencies()[np.argmax(yw_model.psd)]
    if f0 > 1000:
        f0 = np.mean(f0s_burg)
    f0s_yw.append(f0)
    burg_model = pburg(y_sample, order = order, NFFT=NFFT, sampling=sr, scale_by_freq=False)
    f0 = burg_model.frequencies()[np.argmax(burg_model.psd)]
    if f0 > 1000:
        f0 = np.mean(f0s_burg)
    f0s_burg.append(f0)
    if i == N*81:
        fs = np.array([burg_model.frequencies(), yw_model.frequencies()])
        Powers = np.array([burg_model.psd, yw_model.psd])
        labels = np.array([f'Burg (order={order}, NFFT={NFFT})', f'Yule-Walker (order={order}, NFFT={NFFT})'])
        title = 'Parametric PSD'
        plot_PSD(fs, Powers, title, labels)
    
f0s_spectro(D, sr, np.array([time_p, time_p]), np.array([f0s_burg, f0s_yw ]), 
            ['Burg', 'Yule-Walker'], ax_title='Parametric f0s', has_spectro=False)



#%% MUSIC & Eigenvalue (EV) & Minimum Variance (EV)

# music
f0s_mv = []; f0s_music = []; f0s_ev = []
mv_order = 32; IP=32; NSIG=16; NFFT=4096
for i in np.arange(0, len(y)-N, N): 
    y_sample = y[i:i+N]
    mv_model = pminvar(y_sample, mv_order, NFFT=4096, sampling=sr)
    music_model = pmusic(y_sample, IP=IP, NSIG=NSIG, NFFT=NFFT, sampling=sr, threshold=None, 
                         criteria='aic', verbose=False, scale_by_freq=False)
    ev_model = pev(y_sample, IP=IP, NSIG=NSIG, NFFT=NFFT, sampling=sr,
       scale_by_freq=False, threshold=None, criteria='aic', verbose=False)
    f0 = mv_model.frequencies()[np.argmax(mv_model.psd)]; f0s_mv.append(f0)
    f0 = music_model.frequencies()[np.argmax(music_model.psd)]; f0s_music.append(f0)
    f0 = ev_model.frequencies()[np.argmax(ev_model.psd)]; f0s_ev.append(f0)
    if i == N*81:
        fs = np.array([mv_model.frequencies(), music_model.frequencies(), ev_model.frequencies()])
        Powers = np.array([mv_model.psd, music_model.psd, ev_model.psd])
        labels = np.array([f'MV(Capon) (Order: {mv_order})', f'MUSIC (IP {IP}, NSIG {NSIG}, NFFT {NFFT})',
                          f'EV (IP {IP}, NSIG {NSIG}, NFFT {NFFT})'])
        title = 'Non-Parametric PSD'
        plot_PSD(fs, Powers, title, labels)
        
f0s_spectro(D, sr, np.array([time_p, time_p, time_p]), np.array([f0s_mv, f0s_music, f0s_ev ]), 
            ['MV', 'MUSIC', 'EV'], ax_title='Parametric f0s', has_spectro=False)        


# ## 4.

#%% Summary
import spectrum
norm = True
sides = 'centerdc'

# Parametric Methods 
fig_param = plt.figure(figsize=(15,10))
plt.title('Prediction Approaches')
# MA method
p = spectrum.pma(y_sample, 16, 30, NFFT=4096)
p(); p.plot(label='MA (16, 30)', norm=norm, sides=sides)

# ARMA method
p = spectrum.parma(y_sample, 16, 16, 30, NFFT=4096)
p(); p.plot(label='ARMA(16,16)', norm=norm, sides=sides)

# yulewalker
p = spectrum.pyule(y_sample, 16, norm='biased', NFFT=4096)
p(); p.plot(label='YuleWalker(16)', norm=norm, sides=sides)

#burg method
p = spectrum.pburg(y_sample, order=16, NFFT=4096)
p(); p.plot(label='Burg(16)', norm=norm, sides=sides)


plt.legend(loc='upper left', prop={'size':10}, ncol=2)
plt.ylim([-80,10])


#%% Non-Parmaetric Methods
fig_non_param = plt.figure(figsize=(15,10)) 
plt.title('Non-Parametric Approaches')
# correlagram
p = spectrum.pcorrelogram(y_sample, lag=16, NFFT=4096)
p(); p.plot(label='Correlogram(16)', norm=norm, sides=sides)

# minvar
p = spectrum.pminvar(y_sample, 16, NFFT=4096)
p(); p.plot(label='minvar (16)', norm=norm, sides=sides)

# music
p = spectrum.pmusic(y_sample, 16, 11, NFFT=4096)
p(); p.plot(label='MUSIC (16, 11)', norm=norm, sides=sides)

#covar method
p = spectrum.pcovar(y_sample, 16, NFFT=4096)
p(); p.plot(label='Covar(16)', norm=norm, sides=sides)

# modcovar method
p = spectrum.pmodcovar(y_sample, 16, NFFT=4096)
p(); p.plot(label='Modcovar(16)', norm=norm, sides=sides)

plt.legend(loc='upper left', prop={'size':10}, ncol=2)
# plt.ylim([-80,10])

plt.show()