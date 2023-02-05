import scipy.io.wavfile as wav 
import matplotlib.pyplot as plt 
from scipy.fftpack import fft 

# Read the .wav file 
rate, data = wav.read('data_augmented/KillerWhale/KillerWhale_aug_3.wav') 

# Take the FFT of the data 
fft_out = fft(data) 

# Calculate the power spectral density 
psd = abs(fft_out)**2 

# Plot the PSD 
plt.plot(psd) 
plt.title('KillerWhale') 
plt.xlabel('Frequency') 
plt.ylabel('Power Spectral Density') 
plt.show() 