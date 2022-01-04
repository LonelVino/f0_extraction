import numpy as np
from playsound import playsound
from scipy.fft import ifft
from scipy.io.wavfile import write
from tqdm import tqdm 
import matplotlib.pyplot as plt

class Restore:
    '''
    Args
        :params max_dist (float64): the maximal tolerant distance between audio amplitude and average amplitude
                used to filter the outlier
        :params amp_range (int16): the amplitude range of the original audio
                used to rescale the processed audio
        :params fs_rate (int16): sampling rate of the signal, defines the number of samples per second 
    '''
    def __init__(self, amp_range, fs_rate):
        self.amp_range = amp_range
        self.fs_rate = fs_rate
    
    def restore(self, FFT_array):
        '''
        Args
            :params FFT_array (list): FFT value of each frame of signal
            
        Returns
            signal_restored (list): restored signal
        '''
        signal_array = []
        for FFT in tqdm(FFT_array):
            FFT_item = np.concatenate((FFT, FFT[::-1]), axis=None)
            signal = ifft(FFT_item)
            signal_array.append(signal.real)
        signal_restored = np.concatenate(signal_array)
        return signal_restored


    def signal_process(self, signal_restored, signal_original):
        # Process the restore audio: resize outlier value, rescale to the same size
        signal_appr = np.copy(signal_restored)
        # Process the outlier
        # Rescale into original amplitude
        signal_appr *= int(self.amp_range/(max(signal_appr)-min(signal_appr)))
        signal_appr = signal_appr.astype(np.int16)
        return signal_appr
    
    
    def compare_signal(self, total_time, signal_original, signal_restored, signal_processed):
        total_t = np.arange(0, total_time, 1/self.fs_rate)
        plt.figure(figsize=(6,12))
        plt.subplots_adjust(left=0.1, right=0.9, 
                            bottom=0.1, top=0.9, 
                            wspace=0.4, hspace=0.3)

        ax1, ax2, ax3 = plt.subplot(3,1,1), plt.subplot(3,1,2), plt.subplot(3,1,3)
        ax1.plot(total_t[:len(signal_restored)], signal_restored, "b") # plotting the signal
        ax1.set_xlabel('Time'); ax1.set_ylabel('Amplitude')
        ax1.set_title('Raw Restored Audio')
        ax2.plot(total_t[:len(signal_processed)], signal_processed, "b") # plotting the signal
        ax2.set_xlabel('Time'); ax2.set_ylabel('Amplitude')
        ax2.set_title('Processed Restored Audio')
        ax3.plot(total_t, signal_original[:len(total_t)], "b") # plotting the signal
        ax3.set_xlabel('Time'); ax3.set_ylabel('Amplitude')
        ax3.set_title('Orignal Audio')
        
        
    def save_audio(self, signal, audio_path, is_play=False):
        write(audio_path, self.fs_rate, signal)
        if is_play:
            print('\n[INFO]Playing Audio (PATH: %s)'%audio_path)
            playsound(audio_path)