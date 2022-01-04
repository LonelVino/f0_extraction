
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
import scipy
import scipy.fftpack
from scipy.fft import fft, ifft
from tqdm import tqdm

class FFT_f0:
    def __init__(self, time_period=0.1, filename='fluteircam.wav', root='assets/'):
        '''
        Args:
            time_period (float): time window length (s)
            filename (str): audio file name (with eventually the full path)
            root (str): the root path of the directory
        '''
        
        self.time_period = time_period # FFT time window period (in seconds). Can comfortably process time frames from 0.05 seconds - 10 seconds
        self.filename = filename
        self.root = root
                
# %% Initialization 
    def read_audio(self):
        """
        Read an audio file (from scipy.io.wavfile)
        A conversation of the aufio file can be processed using SOX (by default set at False).

        Returns:
            * fs_rate: sampling rate of the signal,defines the number of samples per second
            * signal_original: list of values of the signal
        :rtype: tuple
        """
        print('[INFO] Reading Audio ..... (This may take a while)')
        fs_rate, signal_original = wavfile.read(self.root+self.filename)  # signal_original is a 1-D for 1-channel WAV file

        self.total_time = int(np.floor(len(signal_original)/fs_rate))
        self.sample_range = np.arange(0, self.total_time, self.time_period)
        #TODO: Because the sound pressure values are mapped to integer values that can range from $-2^{15}$ to $(2^{15})-1$. We can convert our sound array to floating point values ranging from $-1$ to $1$ as follows:
        self.signal_original = signal_original / (2.**15)
        self.fs_rate = fs_rate
        
        print("="*80, '\n[AUDIO INFO]')
        print ("Sampling rate of the signal: %d Hz" % fs_rate)
        print ("total time: %.2f s" % self.total_time)
        print ("sample time period: %2.f" % self.time_period)
        print("="*80, '\n')
        
        return fs_rate, self.total_time, signal_original 

# %% Perform FFT on the audio
    def FFT_estimation(self, _signal_):
        FFT_array, freq_array = [], []
        for i in tqdm(self.sample_range):
            sample_start = int(i*self.fs_rate) # the start sample freq
            sample_end = int((i+self.time_period)*self.fs_rate)  # the ending sample freq
            signal = _signal_[sample_start:sample_end]

            N = signal.shape[0]     # number of sampling points = time_period*fs_rate, such as 3200 heres
            
            # ========================== TTF ==========================
            FFT = np.abs(fft(signal)) # Get the FFT 
            FFT_side = FFT[range(int(N/2))] # one side FFT range, abondon the second half
            freqs = scipy.fftpack.fftfreq(N, 1.0/self.fs_rate) # fftfreq(num of sampling points, sampling interval)
            fft_freqs_side = np.array(freqs[range(int(N/2))]) # one side frequency, abondon the second half   

            # FFT Normalize (0-1)
            FFT_side_norm = FFT_side / max(FFT_side) if (max(FFT_side) != 0) else FFT_side

            # Append to output array
            FFT_array.append(FFT_side_norm)
            freq_array.append(fft_freqs_side)
        return FFT_array, freq_array

# %% Visualize local frequency of the whole audio
    def plot_local_freq(self, FFT_array, freq_array, k):
        # Plotting a period of signal, here the last period of signal, e.g. [12.9s, 13.0s]
        # k: the index of the window in the whole audio
        plt.figure(figsize=(8,10))

        total_t = np.arange(0, self.total_time, 1.0/self.fs_rate)
        plt.subplot(211)
        plt.plot(total_t, self.signal_original[:len(total_t)], "b") # plotting the signal
        plt.axvline(x=self.sample_range[k], ymin=0, ymax=1)
        plt.axvline(x=self.sample_range[k-1], ymin=0, ymax=1)
        plt.xlabel('Time'); plt.ylabel('Amplitude')
        plt.title('Time Series (All audio)')


        plt.subplot(212)
        diff = np.diff(freq_array[k])  # Calculate the n-th discrete difference
        widths = np.hstack([diff, diff[-1]]) # Stack arrays in sequence horizontally (column wise).
        plt.bar(freq_array[k], abs(FFT_array[k]), width=widths) # plotting the positive fft spectrum
        plt.xlim(0, 2000)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('FFT Single-sided')
        plt.title('Frequency Series (Last Period)')

        plt.subplots_adjust(left=0.1, bottom=0.1, 
                            right=0.9, top=0.9, 
                            wspace=0.4, hspace=0.4)
    

    def extract_all_f0(self, FFT_array, freq_array):
        '''
        Extract all fundamental frequencies and Visualize by freq-time figure
        
        Args:
            FFT_array (array of complex): the FFT of all windows
            freq_array (array of float): the frequencies of the whole audio
        '''
        x_steps, y_steps = [], []
        for i in range(len(FFT_array)):
            max_freq = freq_array[i][FFT_array[i].argmax()]
            x_steps.append(self.time_period*i)
            y_steps.append(max_freq)

        # Plotting a period of signal, here the last period of signal, e.g. [12.9s, 13.0s]
        plt.figure(figsize=(8,4))
        plt.step(x_steps, y_steps)
        plt.title('Fundamental Frequency')
        plt.xlabel('Time / s')
        plt.ylabel('Frequency / Hz')
        
        f0 = dict(zip(x_steps, y_steps))
        return f0