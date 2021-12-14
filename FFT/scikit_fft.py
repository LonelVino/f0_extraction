
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
import scipy
import scipy.fftpack
from scipy.fft import fft, ifft
from tqdm import tqdm

class FFT_f0:
    def __init__(self, filename='fluteircam.wav', root='assets/'):
        self.filename = filename
        self.root = root
        
# %% Initialization 
    def read_audio(self):
        """
        Read an audio file (from scipy.io.wavfile)
        A conversation of the aufio file can be processed using SOX (by default set at False).

        Args:
            :param filename (str): audio file name (with eventually the full path)
            :param root (str): the root path of the directory
        Returns:
            * fs_rate: sampling rate of the signal,defines the number of samples per second
            * signal_original: list of values of the signal
        :rtype: tuple
        """
        
        fs_rate, signal_original = wavfile.read(self.root+self.filename)  # signal_original is a 1-D for 1-channel WAV file

        self.time_period = 0.1 # FFT time period (in seconds). Can comfortably process time frames from 0.05 seconds - 10 seconds
        self.total_time = int(np.floor(len(signal_original)/fs_rate))
        self.sample_range = np.arange(0, self.total_time, self.time_period)
        self.total_samples = len(self.sample_range)

        print ("Frequency sampling: %d Hz" % fs_rate)
        print(f"number of total freq: {signal_original.shape[0]} Hz")
        print ("total time: %.2f s" % self.total_time)
        print ("sample time period: ", self.time_period)
        print ("total samples: ", self.total_samples)

        #TODO: Because the sound pressure values are mapped to integer values that can range from $-2^{15}$ to $(2^{15})-1$. We can convert our sound array to floating point values ranging from $-1$ to $1$ as follows:
        self.signal_original = signal_original / (2.**15)
        self.fs_rate = fs_rate
        
        return fs_rate, signal_original

# %% Perform FFT on the audio
    def FFT_estimation(self):
        FFT_array, FFT_bucket_array = [], []
        freq_array, freq_bucket_array = [], []
        fs_rate = self.fs_rate
        for i in tqdm(self.sample_range):

            sample_start = int(i*fs_rate) # the start sample freq
            sample_end = int((i+self.time_period)*fs_rate)  # the ending sample freq
            signal = self.signal_original[sample_start:sample_end]

            l_audio = len(signal.shape)  # dimensions (num of channels) of signal
            if l_audio == 2:
                signal = signal.sum(axis=1) / 2
            N = signal.shape[0]     # number of sampling points = time_period*fs_rate, such as 3200 heres

            secs = N / float(fs_rate)   # the sample time period, such as 3200/32000Hz = 0.1s here
            Ts = 1.0/fs_rate   # sampling interval in time
            t = scipy.arange(0, secs, Ts) # time vector as scipy arange field / numpy.ndarray

            # ========================== TTF ==========================
            FFT = abs(fft(signal)) # Get the FFT 
            FFT_side = FFT[range(int(N/2))] # one side FFT range, abondon the second half
            freqs = scipy.fftpack.fftfreq(N, Ts) # fftfreq(num of sampling points, sampling interval)
            fft_freqs_side = np.array(freqs[range(int(N/2))]) # one side frequency, abondon the second half 

            # Reduce to 0-5000 Hz
            bucket_size = 5
            buckets = 16

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


        # Plotting a period of signal, here the last period of signal, e.g. [12.9s, 13.0s]
        plt.figure(figsize=(8,20))

        total_t = np.arange(0, self.total_time, 1.0/fs_rate)
        plt.subplot(411)
        plt.plot(total_t, self.signal_original[:len(total_t)], "b") # plotting the signal
        plt.axvline(x=self.sample_range[-1], ymin=0, ymax=1)
        plt.axvline(x=self.sample_range[-2], ymin=0, ymax=1)
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.title('Time Series (All audio)')

        plt.subplot(412)
        plt.plot(t, signal, "g") # plotting the signal
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.title('Time Series (Last period of audio)')

        plt.subplot(413)
        diff = np.diff(fft_freqs_side)  # Calculate the n-th discrete difference
        widths = np.hstack([diff, diff[-1]]) # Stack arrays in sequence horizontally (column wise).
        plt.bar(fft_freqs_side, abs(FFT_side_norm), width=widths) # plotting the positive fft spectrum
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Count single-sided')
        plt.title('Frequency Series (FFT)')

        plt.subplot(414)
        diff = np.diff(fft_freqs_side_bucket)
        widths = np.hstack([diff, diff[-1]])
        plt.bar(fft_freqs_side_bucket, abs(FFT_side_bucket_norm), width=widths) # plotting the positive fft spectrum
        plt.xticks(fft_freqs_side_bucket, fft_freqs_side_bucket, rotation='vertical')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Count single-sided')
        plt.title('Frequency Series (FFT) (bucket)')

        plt.subplots_adjust(left=0.1,
                            bottom=0.1, 
                            right=0.9, 
                            top=0.9, 
                            wspace=0.4, 
                            hspace=0.4)
        plt.show()

        return FFT_array
    
# %% Extract all fundamental frequencies and Visualize by freq-time figure 
    def plot_all_f0(self, FFT_array):
        len(FFT_array), len(FFT_array[0])

        x_steps, y_steps = [], []
        for i in range(len(FFT_array)):
            max_freq = FFT_array[i].argmax()
            x_steps.append(self.time_period*i)
            y_steps.append(max_freq)

        # Plotting a period of signal, here the last period of signal, e.g. [12.9s, 13.0s]
        plt.figure(figsize=(8,4))
        plt.step(x_steps, y_steps)
        plt.title('Fundamental Frequency')
        plt.xlabel('Time / s')
        plt.ylabel('Frequency / Hz')
        plt.show()
        


# %% Main Part
if __name__ == '__main__':
    filename = 'fluteircam.wav'
    root = 'assets/'
    f0 = FFT_f0(filename, root)
    fs_rate, signal_original = f0.read_audio()
    FFT_array = f0.FFT_estimation()
    f0.plot_all_f0(FFT_array)