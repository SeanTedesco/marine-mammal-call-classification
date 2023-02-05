import os 
from scipy.io import wavfile 
import matplotlib.pyplot as plt  
import numpy as np 

#directory = 'data_augmented/KillerWhale/' 
#directory = 'data_marine_raw/Fin_FinBackWhale/' 
directory = 'data_marine_raw/KillerWhale/' 

for filename in os.listdir(directory): 
    if filename.endswith('.wav'): 
        # Read the .wav file 
        rate, data = wavfile.read(os.path.join(directory, filename)) 
         
        # Perform a Fast Fourier Transform (FFT) on the data 
        fft_data = np.fft.fft(data) 
        fft_data = fft_data[0:len(fft_data)//2] 
         
        # Compute the spectral density by taking the absolute value of the FFT 
        spectral_density = np.abs(fft_data)

        # Plot the spectral density 
        plt.plot(spectral_density, label=filename) 
plt.title('KillerWhale') 
plt.xlabel('Frequency') 
plt.ylabel('Spectral Density') 
plt.legend() 
plt.show() 

 