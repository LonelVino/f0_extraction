from scikit_fft import FFT_f0
from restore import Restore
import matplotlib.pyplot as plt 

# %% Main Part
if __name__ == '__main__':
    filename = 'fluteircam.wav'; root = 'assets/audio/'
    window_period = 0.1
    f0 = FFT_f0(window_period, filename, root)
    
    print('\n', '_'*80, '\n[INFO] Read Audio and Perform FFT', )
    fs_rate, total_time, signal_original = f0.read_audio()
    FFT_array, freq_array = f0.FFT_estimation(signal_original)
    f0.plot_local_freq(FFT_array, freq_array, -20)
    ffs = f0.extract_all_f0(FFT_array, freq_array)  # fundamental frequencies
    plt.show()

    amp_range = abs(max(signal_original) - min(signal_original))
    save_path = root + 'restored/' + filename
    Restore_model = Restore(amp_range, fs_rate)
    
    print('\n', '_'*80, '\n[INFO] Start Restoring Audio From FFT...')
    signal_restored = Restore_model.restore(FFT_array)
    signal_processed = Restore_model.signal_process(signal_restored, signal_original)
    Restore_model.compare_signal(total_time, signal_original, signal_restored, signal_processed)
    Restore_model.save_audio(signal_processed, save_path, is_play=False)
    plt.show()
    