
import os
import librosa
import math
import json


SAMPLE_RATE = 22050
DURATION = 30 # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    
    # dictionary to store data
    data = {
        "mapping": [],
        "mfcc": [],
        "labels": [],
    }
    
    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length) # 1.2 -> 2
    
    # loop through all the animals
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        
        # ensure that we're not at the root level
        if dirpath is not dataset_path:
            
            # save the semantic label
            dirpath_components = dirpath.split("/") # animals/killerwhale => ['animals', 'killerwhale', 'humpback']
            semantic_label = dirpath_components[-1]
            if semantic_label not in ('.ipynb_checkpoints'):
                data['mapping'].append(semantic_label)
                print("\nProcessing {}".format(semantic_label))
            
            
            # process files for a specific marine animal
            for f in filenames:
                
                # load the audio file
                file_path = os.path.join(dirpath, f)
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                
                # process segments extracting mfcc and storing the data
                for s in range(num_segments):
                    start_sample = num_samples_per_segment * s # s =0 -> 0
                    finish_sample = start_sample + num_samples_per_segment # s=0 -> num_samples_per_segment
                    
                    mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample],
                                               sr=sr, 
                                               n_fft=n_fft,
                                               n_mfcc=n_mfcc,
                                               hop_length=hop_length)
                    
                    mfcc = mfcc.T
                    
                    # store mfcc for segment if it has the expected length 
                    if len(mfcc) == expected_num_mfcc_vectors_per_segment: #  and semantic_label not in ('.ipynb_checkpoints')
                        data['mfcc'].append(mfcc.tolist())
                        data['labels'].append(i)
                        print("{}, segment:{}".format(file_path, s+1))
                    
                
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)