#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from scipy.io.wavfile import read as wavread
from os import sep
import time
import logging


def audio_read(audioFilePath):
    """
    Read an audio file (from scipy.io.wavfile)
    A conversation of the aufio file can be processed using SOX (by default set at False).

    Args:
        :param audioFilePath (str): audio file name (with eventually the full path)

    Returns:
        * sr: sampling rate of the signal,defines the number of samples per second
        * sig: list of values of the signal
    :rtype: tuple
    """
    
    logging.info('Reading of the audio file : ' + audioFilePath)
    [sr, sig] = wavread(audioFilePath)
    return sr, sig


def differenceFunction_original(x, N, tau_max):
    """
    Compute difference function of data x. This corresponds to equation (10) in [1.2]

    Args:
        :param x: audio data
        :param N: length of data
        :param tau_max: integration window size
    
    Returns 
        :rtype: list
    """
    df = [0] * tau_max  # initialize the differential window
    for tau in range(1, tau_max):
         for j in range(0, N - tau_max):
             tmp = float(x[j] - x[j + tau])
             df[tau] += tmp**2
    return df


def differenceFunction_scipy(x, N, tau_max):
    """
    Compute difference function of data x. This corresponds to equation (10) in [1.2]

    Faster implementation of the difference function.
    The required calculation can be easily evaluated by Autocorrelation function or similarly by convolution.
    Wiener–Khinchin theorem allows computing the autocorrelation with two Fast Fourier transforms (FFT), with time complexity O(n log n).
    This function use an accellerated convolution function `fftconvolve` from Scipy package.

    Args:
        :param x: audio data
        :param N: length of data
        :param tau_max: integration window size
    
    Returns
        :rtype: list
    """
    x = np.array(x, np.float64)
    w = x.size
    # `cumsum`: Return the cumulative sum of the elements along a given axis
    x_cumsum = np.concatenate((np.array([0]), (x * x).cumsum()))
    # scipy.signal.fftconvolve(in1, in2, mode='full', axes=None)
    # The algorithm is same as convolve, but faster
    # Convolve in1 and in2 using the fast Fourier transform method, with the output size determined by the mode argument.
    conv = fftconvolve(x, x[::-1])  # length: 2*w-1
    tmp = x_cumsum[w:0:-1] + x_cumsum[w] - x_cumsum[:w] - 2 * conv[w - 1:]
    return tmp[:tau_max + 1]




def differenceFunction(x, N, tau_max):
    """
    Compute difference function of data x. This corresponds to equation (10) in [1.2]

    Fastest implementation. Use the same approach as differenceFunction_scipy.
    This solution is implemented directly with Numpy fft.

    Args:
        :param x: audio data
        :param N: length of data
        :param tau_max: integration window size
    Returns
        :rtype: list
    """

    x = np.array(x, np.float64)
    w = x.size
    tau_max = min(tau_max, w)
    x_cumsum = np.concatenate((np.array([0.]), (x * x).cumsum()))
    size = w + tau_max
    # x.bit_length() is the unique positive integer k such that 2**(k-1) <= abs(x) < 2**k
    p2 = (size // 32).bit_length()
    nice_numbers = (16, 18, 20, 24, 25, 27, 30, 32)
    size_pad = min(x*2**p2 for x in nice_numbers if x*2**p2 >= size)
    fc = np.fft.rfft(x, size_pad)  # FFT for real input.
    conv = np.fft.irfft(fc * fc.conjugate())[:tau_max]
    return x_cumsum[w:w - tau_max:-1] + x_cumsum[w] - x_cumsum[:tau_max] - 2*conv



def cumulativeMeanNormalizedDifferenceFunction(df, N):
    """
    Compute cumulative mean normalized difference function (CMND). This corresponds to equation (12) in [1.3]

    Args:
        :param df: Difference function
        :param N: length of data
    
    Returns cumulative mean normalized difference function
    :rtype: list
    """

    cmndf = df[1:] * range(1, N) / np.cumsum(df[1:]).astype(float) #scipy method
    # numpy.insert(arr, obj, values, axis=None)
    # arr: input array; obj: index or indice; values: Values to insert into arr
    return np.insert(cmndf, 0, 1)  # insert 1 at the beginning



def getPitch(cmdf, tau_min, tau_max, harmo_th=0.1):
    """
    Return fundamental period of a FRAME based on CMND function 
            if there is difference(cmnd) under threshold, 0 otherwise

    Args:
        :param cmdf: Cumulative Mean Normalized Difference function
        :param tau_min: minimum period for speech
        :param tau_max: maximum period for speech
        :param harmo_th: harmonicity threshold to determine if it is necessary to compute pitch frequency
    Returns: 
    :rtype: float
    """
    tau = tau_min
    while tau < tau_max:
        if cmdf[tau] < harmo_th:
            while tau + 1 < tau_max and cmdf[tau + 1] < cmdf[tau]:
                tau += 1
            return tau  # when find the minimal diffenrence
        tau += 1  # if difference larger than threshold

    return 0    # if unvoiced



def compute_yin(sig, sr, dataFileName=None, w_len=512, w_step=256, f0_min=100, f0_max=500, harmo_thresh=0.1):
    """

    Compute the Yin Algorithm. Return fundamental frequency and harmonic rate.

    Args:
        :param sig: Audio signal (list of float)
        :param sr: sampling rate (int)
        :param w_len: size of the analysis window (samples)
        :param w_step: size of the lag between two consecutives windows (samples)
        :param f0_min: Minimum fundamental frequency that can be detected (hertz)
        :param f0_max: Maximum fundamental frequency that can be detected (hertz)
        :param harmo_tresh: Threshold of detection. The yalgorithmù return the first minimum of the CMND fubction below this treshold.

    Returns:

        * pitches: list of fundamental frequencies,
        * harmonic_rates: list of harmonic rate values for each fundamental frequency value (= confidence value)
        * argmins: minimums of the Cumulative Mean Normalized DifferenceFunction
        * times: list of time of each estimation
    :rtype: tuple
    """

    tau_min = int(sr / f0_max)  # the minimal lag = number of samples * minimal sampling period 
    tau_max = int(sr / f0_min)  # the maximal lag = number of samples * maximal sampling period

    timeScale = range(0, len(sig) - w_len, w_step)  # time values for each analysis window
    times = [t/float(sr) for t in timeScale]
    frames = [sig[t:t + w_len] for t in timeScale]

    pitches = [0.0] * len(timeScale)
    harmonic_rates = [0.0] * len(timeScale)
    argmins = [0.0] * len(timeScale)

    for i, frame in enumerate(frames):

        #Compute YIN
        df = differenceFunction(frame, w_len, tau_max)
        cmdf = cumulativeMeanNormalizedDifferenceFunction(df, tau_max)
        p = getPitch(cmdf, tau_min, tau_max, harmo_thresh)

        #Get results
        if np.argmin(cmdf)>tau_min:
            argmins[i] = float(sr / np.argmin(cmdf))
        if p != 0: # A pitch was found
            pitches[i] = float(sr / p)  # Pitch = sampling number (times) * (1/fundamental period)
            harmonic_rates[i] = cmdf[p]
        else: # No pitch, but we compute a value of the harmonic rate
            harmonic_rates[i] = min(cmdf)


    if dataFileName is not None:
        np.savez(dataFileName, times=times, sr=sr, w_len=w_len, w_step=w_step, f0_min=f0_min, f0_max=f0_max, harmo_thresh=harmo_thresh, pitches=pitches, harmonic_rates=harmonic_rates, argmins=argmins)
        print('\t- Data file written in: ' + dataFileName)

    return pitches, harmonic_rates, argmins, times


def main(audioFileName="fluteircam.wav", w_len=1024, w_step=256, f0_min=70, f0_max=200,\
         harmo_thresh=0.85, audioDir="assets", dataFileName=None, verbose=4):
    """
    Run the computation of the Yin algorithm on a example file.
    Write the results (pitches, harmonic rates, parameters ) in a numpy file.

    Args:
        :param audioFileName (str): name of the audio file
        :param w_len (int): length of the window
        :param w_step (int): length of the "hop" size 
        :param f0_min (float): minimum f0 in Hertz
        :param f0_max (float): maximum f0 in Hertz 
        :param harmo_thresh (float): harmonic threshold 
        :param audioDir (str): path of the directory containing the audio file
        :param dataFileName (str): file name to output numpy results 
        :param verbose (int): Outputs on the console : 0-> nothing, 1-> warning, 2 -> info, 3-> debug(all info), 4 -> plot + all info
    """

    if audioDir is not None:
        audioFilePath = audioDir + sep + audioFileName
    else:
        audioFilePath = audioFileName

    sr, sig = audio_read(audioFilePath)

    start = time.time()
    pitches, harmonic_rates, argmins, times = compute_yin(sig, sr, dataFileName, w_len, w_step, f0_min, f0_max, harmo_thresh)
    end = time.time()
    print("[INFO] Yin computed in: %.4f seconds"%(end - start))

    duration = len(sig)/float(sr)
    
    if verbose > 3:
        fig = plt.gcf()
        fig.set_size_inches(10.5, 18.5)
        fig.savefig('pics/YIN.png', dpi=100)
        ax1 = plt.subplot(4, 1, 1)
        ax1.plot([float(x) * duration / len(sig) for x in range(0, len(sig))], sig)
        ax1.set_title('Audio data'); ax1.set_ylabel('Amplitude')
        ax2 = plt.subplot(4, 1, 2)
        ax2.plot([float(x) * duration / len(pitches) for x in range(0, len(pitches))], pitches)
        ax2.set_title('F0'); ax2.set_ylabel('Frequency (Hz)')
        ax3 = plt.subplot(4, 1, 3, sharex=ax2)
        ax3.plot([float(x) * duration / len(harmonic_rates) for x in range(0, len(harmonic_rates))], harmonic_rates)
        ax3.plot([float(x) * duration / len(harmonic_rates) for x in range(0, len(harmonic_rates))], [harmo_thresh] * len(harmonic_rates), 'r')
        ax3.set_title('Harmonic rate'); ax3.set_ylabel('Rate')
        ax4 = plt.subplot(4, 1, 4, sharex=ax2)
        ax4.plot([float(x) * duration / len(argmins) for x in range(0, len(argmins))], argmins)
        ax4.set_title('Index of minimums of CMND')
        ax4.set_ylabel('Frequency (Hz)'); ax4.set_xlabel('Time (seconds)')
        
        plt.subplots_adjust(left=0.1, bottom=0.1, 
                    right=0.9, top=0.9, 
                    wspace=0.4, hspace=0.4)
        plt.show()
        
    # Shut down the logger
    logging.shutdown()



if __name__ == '__main__':
    main()


