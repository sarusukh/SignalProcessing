import scipy.io

fname = './sunspot.dat'
spdata = scipy.io.loadmat(fname)

year = spdata.get('sunspot')[:,0]
wolfer = spdata.get('sunspot')[:,1]

import pylab

figid = pylab.figure(num=1,figsize=(15,6))

pn = figid.add_subplot(1,1,1)
pn.plot(year, wolfer )
pylab.title('Sunspot Data'); pylab.xlabel('Year'); 
pylab.ylabel('Wolfer Index')

# FFT of sunspot data
Y = pylab.fft(wolfer)
Y[0] = complex( pylab.nan, pylab.nan )

figid = pylab.figure(num=2,figsize=(8,8))
pn = figid.add_subplot(1,1,1)
pn.plot(Y.real,Y.imag,linestyle='None',marker='o')
pn.set_title('Fourier Coefficients in the Complex Plane')
pn.set_xlabel('Real Axis'); 
pn.set_ylabel('Imaginary Axis')

n = len(Y)
power = abs(Y[0:(n/2)])**2
nyquist = 1. / 2.
freq = pylab.arange(0, n / 2) / (n / 2.) * nyquist

figid = pylab.figure(num=3, figsize=(8,6))
pn = figid.add_subplot(1,1,1)
pn.plot(freq, power )
pn.set_title('Power Spectrum of Sunspot Data')
pn.set_xlabel('Frequency (cycles/Year)')
pn.set_ylabel('|FFT(f(t))|$^2$')

period = 1. / freq

fig = pylab.figure( figsize=(8,6) )
pylab.plot(period, power)
pylab.xlim( 0., 100. )
pylab.title('Power Spectrum of Sunspot Data')
pylab.xlabel('Period (Years/cycle)')
pylab.ylabel('|FFT(f(t))|^2')

# Find finite values
indFinite = pylab.where( pylab.isfinite(power) )[ 0 ]

# Find index where |F|^2 is maximum
maxpower = power[ indFinite ].max()
ind = pylab.where( power[ indFinite ] == maxpower )[ 0 ]

# Peak value
pylab.plot(period[ indFinite[ ind ] ], power[ indFinite[ ind ] ], marker='o')

# Annotation
pylab.text(period[ indFinite[ ind ] ] + 3, power[ indFinite[ ind ] ], \
        'Period = %8.3f years' % period[ indFinite[ ind ] ], color='k' )
        
        
