
from matplotlib.pyplot import figure, legend, show
from numpy import arange, conj, cos, log10, sin, pi, real
from scipy.fftpack import fft, fftshift

class Wave:

    def __init__(self, fs=500, f0=1., nfft=1024, phase=0.):

        """ Constructor
        """

        # INPUTS

        self.fs = fs            # sampling freq.
        self.f0 = f0            # Initial freq. of the chirp
        self.nfft = nfft        # No. of FFT-points
        self.phase = phase      # Phase shift, in rad

        self.t = arange(0., 1. + 1./self.fs, 1./self.fs) 
        f1 = self.fs / 20.      # Freq. of the chirp at t = 1 s


        # Chirp signal

        t1 = 1.
        T = t1 - self.t[0]
        k = (f1 - self.f0) / T
        self.x = cos(2 * pi * (k/2*self.t + f0) * self.t + phase)

    #
    # End of '__init__'
    ##### 


    def spectrum(self):

        """ 
        """

        # Computes DFT using FFT`
        self.X = fft(self.x, self.nfft)

        # DFT sample points
        self.s_points = arange(self.nfft)

        # FFT shifting
        self.Xs = fftshift(self.X)

        # Normalized Freq.
        self.freq_n = arange(-self.nfft/2., self.nfft/2.) / self.nfft 

        # Power at each freq. component
        self.Px = self.Xs * conj(self.Xs) / (self.nfft * len(self.x))

        # One-sided quantities
        #
        # freq. bins
        self.freq = self.freq_n[int(self.nfft/2)::] * self.fs
        #
        # PSD
        self.PSD = self.Px[int(self.nfft/2)::]

    #
    # End of 'spectrum'
    #####


    def plot(self):

        """ Plot signal and spectrum
        """

        f = figure(figsize=(12,12))

        # Signal
        pn = f.add_subplot(221)
        pn.plot(self.t, self.x)
        pn.set_xlim([0., self.t[-1]])
        pn.set_title('Upchirp Wave')
        pn.set_xlabel('Time(s)')
        pn.set_ylabel('Amplitude')

        # Raw values of DFT
        pn = f.add_subplot(222)
        pn.plot(self.s_points, abs(self.X))
        pn.set_xlim([0, self.nfft-1])
        pn.set_title('Double sided FFT - no Shift')
        pn.set_xlabel('Sample points (N-point DFT)')
        pn.set_ylabel('DFT Values')

        # Raw values vs Normalized freq.
        pn = f.add_subplot(223)
        pn.plot(self.freq_n, abs(self.Xs))
        pn.set_xlim([-.1, .1])
        pn.set_title('Double sided FFT')
        pn.set_xlabel('Normalized freq.')
        pn.set_ylabel('DFT Values')

        # One-sided Power Spectral Density (PSD)
        pn = f.add_subplot(224)
        pn.plot(self.freq, 10*log10(real(self.PSD)))
        pn.set_title('Power Spectral Density')
        pn.set_xlabel('Frequency (Hz)')
        pn.set_ylabel('Power')

    #
    # End of 'plot'
    #####

#
# End of 'Wave' 
#####


if __name__ == '__main__':

    Obj = Wave()
    Obj.spectrum()
    Obj.plot()
    show()

#
# End of 'if'
#####