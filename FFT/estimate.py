import numpy as np
imo

# %% Evaluate f0 estimation
def evaluate(signal_original, noise_std, time_period, thresh_ratio):
    # normal(mean of the normal distribution, standard deviation of the normal distribution, the number of elements in array noise)
    noise = np.random.normal(0,noise_std,len(signal_original))
    signal_eval = noise + signal_original
    FFT_array_eval, freq_array_eval, FFT_bucket_array_eval, freq_bucket_array_eval = f0.FFT_estimation(signal_eval)
    ffs_eval = f0.extract_all_f0(FFT_array_eval)
    
    SNRs = {}
    for i in range(len(ffs)):
        start_idx = int(list(ffs.keys())[i] * fs_rate) 
        end_idx = int(start_idx + time_period * fs_rate)
        if abs(list(ffs.values())[i] - list(ffs_eval.values())[i]) > thresh_ratio*list(ffs.values())[i]:
            P_origin = signal_original[start_idx:end_idx]**2
            P_noise = noise[start_idx:end_idx]**2
            SNR = np.mean(P_origin)/np.mean(P_noise)
            SNRs[list(ffs.keys())[i]] = SNR
            
    return SNRs