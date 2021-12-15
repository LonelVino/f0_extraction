from scikit_fft import FFT_f0
from restore import Restore
import matplotlib.pyplot as plt 

# %% Main Part
if __name__ == '__main__':
    filename = 'fluteircam.wav'; root = 'assets/'
    window_period = 0.1
    f0 = FFT_f0(window_period, filename, root)
    fs_rate, signal_original = f0.read_audio()
    FFT_array, freq_array, FFT_bucket_array, freq_bucket_array = f0.FFT_estimation(signal_original)
    ffs = f0.extract_all_f0(FFT_array, freq_array)  # fundamental frequencies
    plt.show()
    
    max_dist = 6e-3 
    amp_range = abs(max(signal_original) - min(signal_original))
    save_path = root + 'restored_audio.wav'
    Restore_model = Restore(max_dist, amp_range, fs_rate)
    
    print('\n[INFO] Start Restoring Audio From FFT...')
    signal_restored = Restore_model.restore(FFT_array)
    signal_processed = Restore_model.signal_process(signal_restored)
    Restore_model.save_audio(signal_processed, save_path, is_play=True)
    