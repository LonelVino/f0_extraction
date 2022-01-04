# Fundamental Frequency Extraction

![](https://img.shields.io/badge/Python-v3.8-orange) ![](https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white)

A Synthetical project with kinds of methods of fundamental frequency extraction, with YIN. SWIPE. CREPE, SPICE.

- **YIN** (Signal Processing, Time Domain)
- **SWIPE** (Signal Processing, Frequency Domain)
- **CREPE**  (Machine Learning, Time Domain, Supervision)
- **Spice** (Machine Learning, Frequency Domain, Self-Supervision)

## Preview
The pitch is one of the three major properties of the sound (volume, sound, tone, tone). And the pitch is determined by the Fundamental Frequency $f_0$.

**Pitch Estimation** (fundamental frequency extraction) has a wide range of applications in sound processing. Typically we first puts the signal into fragment, and then extracts $f_0$ frame by frame.

The method of extracting can be generally divided into **time domain method** and **frequency domain method**:

- Time domain method: The input is the waveform of sound; The basic  principle is to find the **minimum positive cycle** of waveforms.
- Frequency domain method: First perform <u>Fourier transform</u> to obtain a spectrum of signal (with amplitude spectrum, discard the phase spectrum). There will be spikes at $f_0$ in the spectrum, so the basic principle is to require the **greatest common divisor** of these spikes.

## Prerequisites

 * [Numpy](http://www.numpy.org/)
 * [Scipy](http://www.scipy.org/)
 * [Matlplotlib](http://matplotlib.org/) (for graphing)

```bash
git clone git@github.com:LonelVino/f0_extraction.git
cd f0_extraction
pip install -r requirements.txt
```

## Usage
- [YIN Manually](./YIN/README.md)

## Reference

1. ^A. de Cheveigné and H. Kawahara, "YIN, a fundamental frequency estimator for speech and music", Journal of the Acoustical Society of America, 2002.
2. ^M. J. Ross, et al., "Average magnitude difference function pitch estimator", IEEE Transactions on Acoustics, Speech, and Signal Processing, 1974.
3. ^M. Mauch and S. Dixon, "pYIN: A fundamental frequency estimator using probabilistic threshold distributions", ICASSP, 2014.
4. ^A. Camacho and J. G. Harris, "A sawtooth waveform inspired pitch estimator for speech and music", Journal of the Acoustical Society of America, 2008.
5. ^D. J. Hermes, "Measurement of pitch by subharmonic summation", Journal of the Acoustical Society of America, 1988.
6. ^J. W. Kim, et al., "CREPE: A convolutional representation for pitch estimation", ICASSP, 2018.
7. ^B. Gfeller, et al., "SPICE: Self-supervised pitch estimation", IEEE Transactions on Audio, Speech and Language Processing, 2020.
8. ^abM. Tagliasacchi, "SPICE: Self-supervised pitch estimation", Google AI Blog, 2019. Online: https://ai.googleblog.com/2019/11/