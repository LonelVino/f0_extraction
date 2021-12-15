
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
            :param time_period (float): time window length (s)
            :param filename (str): audio file name (with eventually the full path)
            :param root (str): the root path of the directory
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
        
        fs_rate, signal_original = wavfile.read(self.root+self.filename)  # signal_original is a 1-D for 1-channel WAV file

        self.total_time = int(np.floor(len(signal_original)/fs_rate))
        self.sample_range = np.arange(0, self.total_time, self.time_period)
        #TODO: Because the sound pressure values are mapped to integer values that can range from $-2^{15}$ to $(2^{15})-1$. We can convert our sound array to floating point values ranging from $-1$ to $1$ as follows:
        self.signal_original = signal_original / (2.**15)
        self.fs_rate = fs_rate
        
        print('[AUDIO INFO]\n', "="*80)
        print ("Sampling rate of the signal: %d Hz" % fs_rate)
        print ("total time: %.2f s" % self.total_time)
        print ("sample time period: %2.f" % self.time_period)
        print("="*80, '\n')
        
        return fs_rate, signal_original

# %% Perform FFT on the audio
    def FFT_estimation(self, _signal_):
        FFT_array, FFT_bucket_array = [], []
        freq_array, freq_bucket_array = [], []
        fs_rate = self.fs_rate
        for i in tqdm(self.sample_range):

            sample_start = int(i*fs_rate) # the start sample freq
            sample_end = int((i+self.time_period)*fs_rate)  # the ending sample freq
            signal = _signal_[sample_start:sample_end]

            l_audio = len(signal.shape)  # dimensions (num of channels) of signal
            if l_audio == 2:
                signal = signal.sum(axis=1) / 2
            N = signal.shape[0]     # number of sampling points = time_period*fs_rate, such as 3200 heres

            secs = N / float(fs_rate)   # the sample time period, such as 3200/32000Hz = 0.1s here
            Ts = 1.0/(fs_rate*self.time_period)   # sampling interval in time
            t = scipy.arange(0, secs, Ts) # time vector as scipy arange field / numpy.ndarray

            # ========================== TTF ==========================
            FFT = np.real(fft(signal)) # Get the FFT #TODO: if use absolute value or not 
            FFT_side = FFT[range(int(N/2))] # one side FFT range, abondon the second half
            freqs = scipy.fftpack.fftfreq(N, Ts) # fftfreq (num of sampling points, sampling interval)
            fft_freqs_side = np.array(freqs[range(int(N/2))]) # one side frequency, abondon the second half 

            # Reduce to 0-5000 Hz
            bucket_size = 5; buckets = 16
            FFT_side_bucket = FFT_side[0:bucket_size*buckets]
            fft_freqs_side_bucket = fft_freqs_side[0:bucket_size*buckets]
            
            # Combine frequencies into buckets
            FFT_side_bucket = np.array([int(sum(FFT_side_bucket[current: current+bucket_size])) for current in range(0, len(FFT_side_bucket), bucket_size)])
            fft_freqs_side_bucket = np.array([int(sum(fft_freqs_side_bucket[current: current+bucket_size])) for current in range(0, len(fft_freqs_side_bucket), bucket_size)])
            
            # FFT Normalize (0-1)
            FFT_side_norm = FFT_side / max(FFT_side) if (max(FFT_side) != 0) else FFT_side
            FFT_side_bucket_norm = FFT_side_bucket / max(FFT_side_bucket) if (max(FFT_side_bucket) != 0) else FFT_side_bucket
                
            # Append to output array
            FFT_array.append(FFT_side_norm)
            freq_array.append(fft_freqs_side)
            FFT_bucket_array.append(FFT_side_bucket_norm)
            freq_bucket_array.append(fft_freqs_side_bucket)
        
        return FFT_array, freq_array, FFT_bucket_array, freq_bucket_array

# %% Visualize local frequency of the whole audio
    def plot_local_freq(self, FFT_array, freq_array, FFT_bucket_array, freq_bucket_array, k):
        # Plotting a period of signal, here the last period of signal, e.g. [12.9s, 13.0s]
        plt.figure(figsize=(8,20))

        total_t = np.arange(0, self.total_time, 1.0/self.fs_rate)
        plt.subplot(311)
        plt.plot(total_t, self.signal_original[:len(total_t)], "b") # plotting the signal
        plt.axvline(x=self.sample_range[k], ymin=0, ymax=1)
        plt.axvline(x=self.sample_range[k-1], ymin=0, ymax=1)
        plt.xlabel('Time'); plt.ylabel('Amplitude')
        plt.title('Time Series (All audio)')


        plt.subplot(312)
        diff = np.diff(freq_array[k])  # Calculate the n-th discrete difference
        widths = np.hstack([diff, diff[-1]]) # Stack arrays in sequence horizontally (column wise).
        plt.bar(freq_array[k], abs(FFT_array[k]), width=widths) # plotting the positive fft spectrum
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('FFT Single-sided')
        plt.title('Frequency Series (Last Period)')

        plt.subplot(313)
        diff = np.diff(freq_bucket_array[k])
        widths = np.hstack([diff, diff[-1]])
        plt.bar(freq_bucket_array[k], abs(FFT_bucket_array[k]), width=widths) # plotting the positive fft spectrum
        plt.xticks(freq_bucket_array[k], freq_bucket_array[k], rotation='vertical')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('FFT single-sided')
        plt.title('Frequency Series (FFT) (bucket)')

        plt.subplots_adjust(left=0.1,
                            bottom=0.1, 
                            right=0.9, 
                            top=0.9, 
                            wspace=0.4, 
                            hspace=0.4)
        plt.show()
    
# %% Extract all fundamental frequencies and Visualize by freq-time figure 
    def extract_all_f0(self, FFT_array, freq_array):
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
    
# %% Restore the Signal    
    def restore(FFT_array):
        signal_array = []
        for FFT in FFT_array:
            FFT_item = np.concatenate((FFT, FFT[::-1]), axis=None)
            signal = ifft(FFT_item)
            signal_array.append(signal)
        return signal_array
