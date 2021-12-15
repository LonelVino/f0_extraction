import numpy as np
from playsound import playsound
from scipy.fft import ifft
from scipy.io.wavfile import write
from tqdm import tqdm 

class Restore:
    '''
    Args
        :params max_dist (float64): the maximal tolerant distance between audio amplitude and average amplitude
                used to filter the outlier
        :params amp_range (int16): the amplitude range of the original audio
                used to rescale the processed audio
        :params fs_rate (int16): sampling rate of the signal, defines the number of samples per second 
    '''
    def __init__(self, max_dist, amp_range, fs_rate):
        self.max_dist = max_dist
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


    def nearest_approximate(self, signal_, outlier, std, approx_num):
        signal = np.copy(signal_)
        idx = np.where(outlier)[0]
        for i in idx:
            gauss = np.random.normal(0,std)
            appr = np.mean(abs(signal[i-approx_num:i])) if i-approx_num >= 0 else np.mean(abs(signal[:i]))
            signal[i] = appr + gauss
        return signal


    def signal_process(self, signal_restored, approx_num=50):
        '''
        Process the restore audio: resize outlier value, rescale to the same size
        '''
        mean = np.mean(signal_restored[:])
        standard_deviation = np.std(signal_restored)
        distance_from_mean = abs(signal_restored - mean)
        outlier = distance_from_mean > self.max_dist
        signal_appr = self.nearest_approximate(signal_restored, outlier, std=standard_deviation, approx_num=approx_num)
        signal_appr *= int(self.amp_range/(max(signal_appr)-min(signal_appr)))
        signal_appr = signal_appr.astype(np.int16)
        return signal_appr
    
    def save_audio(self, signal, audio_path, is_play=False):
        write(audio_path, self.fs_rate, signal)
        if is_play:
            print('\n[INFO]Playing Audio (PATH: %s)'%audio_path)
            playsound(audio_path)