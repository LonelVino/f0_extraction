
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

mpl.rcParams['font.size'] = 18.0
mpl.rcParams['axes.titlesize'] = 18.0; mpl.rcParams['axes.labelsize'] = 18.0
mpl.rcParams["figure.facecolor"] = 'darkslategray'
plt.style.use('dark_background')
plt.rcParams['axes.facecolor'] = 'darkslategray'
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['xtick.labelsize'] = 'xx-large'
plt.rc('xtick', labelsize='x-large'); plt.rc('ytick', labelsize='x-large')
plt.rc('axes', labelsize='x-large', titlesize='x-large')


# %%
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

# %%
def f0s_spectro(spectro, sr, time_data, f0s_data, labels, ax_title, has_spectro=True):
    fig, ax = plt.subplots(nrows=1, figsize=(12,6))
    if has_spectro:
        img = librosa.display.specshow(spectro, x_axis='time', y_axis='log', ax=ax, sr=sr)
        fig.colorbar(img, ax=ax, format="%+2.f dB")
    ax.set(title=ax_title)
    for idx, f0s in enumerate(f0s_data):
        ax.step(time_data[idx], f0s, label=labels[idx], linewidth=5)
    ax.grid(axis='x', color='0.95')
    ax.legend(title='PSD Estimation Methods')
    ax.set_xlim(0,time_data[0][-1])
    plt.show()

root = './assets/audio/';
file_names = ["croisement.wav", "croisement2.wav", "croisement3.wav", "croisement4.wav"]


# %%
for file_name in file_names:
    # Read the Audiofile
    sr, y = read(root+file_name)
    y = y.astype(float)
    duration = len(y)/sr # # Duration of the audio in Seconds
    time = np.arange(0,duration,1/sr)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)  # Spectrogram in dB

    start = len(y)/4; N = 400;
    time_p = np.arange(0,len(y),N)[:-1]/sr


    #%%2.1 Welch & Daniell 
    from scipy import signal
    # n_overlap: If None, noverlap = nperseg // 2.
    f0s_higher = []; f0s_lower = [];
    nperseg = 1024; NFFT = 4096; order = 32; N =1024
    for i in np.arange(0, len(y)-N, N): 
        y_sample = y[i:i+N]
        f1, psd_lower = signal.welch(y_sample, sr, window='hann', nperseg=nperseg, noverlap=None, 
                            nfft=NFFT, scaling='spectrum')
        higher_model = pev(y_sample, IP=16, NFFT=NFFT, sampling = sr)
        f0_lower = f1[np.argmax(psd_lower)]; f0_higher = higher_model.frequencies()[np.argmax(higher_model.psd)]
        
        f0s_higher.append(f0_higher); f0s_lower.append(f0_lower)

    time_p = np.linspace(0,duration,len(f0s_higher))
    f0s_spectro(D, sr, np.array([time_p, time_p]), np.array([f0s_higher, f0s_lower]), 
                ['Signal 1 (Increase)', 'Signal 2 (Constant)'], ax_title=f'Welch Speration (Non-Parameter {file_name})', has_spectro=True)


    # %%  3.1 Yule-Walker & Burg
    order= 32; NFFT = 4096; N =1500
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
        
    time_p = np.linspace(0,duration,len(f0s_burg))
    f0s_spectro(D, sr, np.array([time_p, time_p]), np.array([f0s_burg, f0s_yw ]), 
                ['Signal 1 (Increase)', 'Signal 2 (Constant)'], ax_title=f'Burg Speration (Parametric, {file_name})', has_spectro=True)



# %% Croisement 4 .wav
file_name =  file_names[3]
# Read the Audiofile
sr, y = read(root+file_name)
y = y.astype(float)
duration = len(y)/sr # # Duration of the audio in Seconds
time = np.arange(0,duration,1/sr)
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)  # Spectrogram in dB


# %% Non-parametric approach
from scipy import signal
# n_overlap: If None, noverlap = nperseg // 2.
f0s_raw = []; f0s_increase = []; f0s_decrease = []; flag = 0
nperseg = 1024; NFFT = 4096; order = 32; N =512
for i in np.arange(0, len(y)-N, N): 
    y_sample = y[i:i+N]
    f1, psd = signal.welch(y_sample, sr, window='hann', nperseg=nperseg, noverlap=None, 
                        nfft=NFFT, scaling='spectrum')
    idx = signal.find_peaks(psd, width=5, distance=10)[0]
    f0 = [f1[i] for i in idx if f1[i]<900] 
    flag = int(i/N) if len(f0)==1 else flag
    f0 = [np.min(f0), np.max(f0)] if len(f0)>2 else [f0[0], f0[0]] if len(f0)==1 else f0
    f0s_raw.append(f0)

f0s_raw = np.array(f0s_raw)
f0_increase = np.append(f0s_raw[:,0][:flag], f0s_raw[:,1][flag:])
f0_decrease = np.append(f0s_raw[:,1][:flag], f0s_raw[:,0][flag:])

time_p = np.linspace(0,duration,len(f0_increase))
f0s_spectro(D, sr, np.array([time_p, time_p]), np.array([f0s_raw[:,0], f0s_raw[:,1]]), 
            ['Peak 1', 'Peak 2'], ax_title=f'Raw f0s {file_name}', has_spectro=True)

time_p = np.linspace(0,duration,len(f0_increase))
f0s_spectro(D, sr, np.array([time_p, time_p]), np.array([f0_increase, f0_decrease]), 
            ['Signal 1 (Increase)', 'Signal 2 (Decrease)'], ax_title=f'Welch Speration (Non-Parameter) {file_name}', has_spectro=True)

# %% Parametric Method
# yule-walker
order= 32; NFFT = 4096; N =512
f0_increase = [] ; f0_decrease = []; f0s_raw = []; 
for i in np.arange(0, len(y)-N, N): 
    y_sample = y[i:i+N]    
    burg_model = pburg(y_sample, order = order, NFFT=NFFT, sampling=sr, scale_by_freq=False)
    idx = signal.find_peaks(burg_model.psd)[0]
    f_burg = burg_model.frequencies()
    f0 = [f_burg[i] for i in idx if f_burg[i]<900] 
    flag = int(i/N) if len(f0)==1 else flag
    f0 = [np.min(f0), np.max(f0)] if len(f0)>2 else [f0[0], f0[0]] if len(f0)==1 else f0
    f0s_raw.append(f0)

f0s_raw = np.array(f0s_raw)

f0_increase = np.append(f0s_raw[:,0][:flag], f0s_raw[:,1][flag:])
f0_decrease = np.append(f0s_raw[:,1][:flag], f0s_raw[:,0][flag:])

time_p = np.linspace(0,duration,len(f0_increase))
f0s_spectro(D, sr, np.array([time_p, time_p]), np.array([f0s_raw[:,0], f0s_raw[:,1]]), 
            ['Peak 1', 'Peak 2'], ax_title=f'Raw f0s {file_name}', has_spectro=True)

time_p = np.linspace(0,duration,len(f0_decrease))
f0s_spectro(D, sr, np.array([time_p, time_p]), np.array([f0_increase, f0_decrease ]), 
            ['Signal 1 (Increase)', 'Signal 2 (Decrease)'], ax_title=f'Burg Speration (Parametric, {file_name})', has_spectro=True)


