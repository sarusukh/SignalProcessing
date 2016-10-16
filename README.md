# Signal Processing

1. [Intro to FFT] (http://nbviewer.jupyter.org/github/rilma/SignalProcessing/blob/master/notebook/IntroFFT.ipynb): FFT of a time-series.

2. [Chirp signal] (https://github.com/rilma/SignalProcessing/blob/master/scripts/chirpwave.py): It generates an upchirp signal and computes its spectrum. Here is an example using a sampling frequency (fs) of 500 Hz and 1024 FFT-points (NFFT):
                  ![Chirp example] (https://github.com/rilma/SignalProcessing/blob/master/scripts/chirpwave.png)
                  
3. [The Many Colors of Noise] (https://github.com/rilma/SignalProcessing/blob/master/scripts/colornoise.py): Simple calculation of the power spectrum of a colored noisy signal (white, pink, red, blue, and violet). By default, it generates 1000 samples with mean value and standarda deviation equal to 0 and 1, respectively. Here is an example of a pink noise signal and its corresponding spectrum:
                  ![Pink Noise Samples] (https://github.com/rilma/SignalProcessing/blob/master/scripts/pink_samples.png)
                  ![Pink Noise Spectrum] (https://github.com/rilma/SignalProcessing/blob/master/scripts/pink_spectrum.png)
