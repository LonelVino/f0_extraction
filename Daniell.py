from spectrum import *
def DaniellPeriodogram(data, P, NFFT=None, detrend='mean', sampling=1.,
                       scale_by_freq=True, window='hamming'):
    r"""Return Daniell's periodogram.

    To reduce fast fluctuations of the spectrum one idea proposed by daniell
    is to average each value with points in its neighboorhood. It's like
    a low filter.
    """
    psd = speriodogram(data, NFFT=NFFT, detrend=detrend, sampling=sampling,
                   scale_by_freq=scale_by_freq, window=window)

    if len(psd) % 2 == 1:
        datatype = 'real'
    else:
        datatype = 'complex'

    N = len(psd)
    _slice = 2 * P + 1
    if datatype == 'real': #must get odd value
        newN = np.ceil(psd.size/float(_slice))
        if newN % 2 == 0:
            newN = psd.size/_slice
    else:
        newN = np.ceil(psd.size/float(_slice))
        if newN % 2 == 1:
            newN = psd.size/_slice

    newpsd = np.zeros(int(newN)) # keep integer division
    for i in range(0, newpsd.size):
        count = 0 #needed to know the number of valid averaged values
        for n in range(i*_slice-P, i*_slice+P+1): #+1 to have P values on each sides
            if n > 0 and n<N: #needed to start the average
                count += 1
                newpsd[i] += psd[n]
        newpsd[i] /= float(count)

    #todo: check this
    if datatype == 'complex':
        freq = np.linspace(0, sampling, len(newpsd))
    else:
        df = 1. / sampling
        freq = np.linspace(0,sampling/2., len(newpsd))
    #psd.refreq(2*psd.size()/A.freq());
    #psd.retime(-1./psd.freq()+1./A.size());

    return newpsd, freq