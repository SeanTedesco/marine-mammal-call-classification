'''
Audio Concatenation Library
===========================

This library provides methods to extract, manipulate
and save audio files from a given directory. Makes
use of the pydub library to save files as AudioSegments.
'''

################################################################################
# IMPORTS
#
import pydub
from pydub import AudioSegment
from pydub import AudioSegment
from pydub.utils import make_chunks
import os 
import random 
from random import choice
os.environ["PATH"] += os.pathsep + '/usr/local/bin'

################################################################################
# FUNCTIONS
#
def get_wavs_from_directory(dirpath:str, n_samples:int) -> list[AudioSegment]:
    '''Generates a playlist of audio segments from the files
        in the provided directory.

    Params:
        dirpath [str]: the path to the directory containing
            audio files.
        n_samples [int]: the number of files to be extracted.
    Raises: 
        None
    Return:
        playlist [list[AudioSegment]]: the list of pydub.AudioSegment
            files extracted from the provided directory.
    '''

    # make sure we have a slash at the end of the directory
    if dirpath[0] != '/':
       dirpath += '/'
    
    # create an empty list which will be returned
    playlist = []
    
    # step through the given directory
    for file in os.listdir(dirpath):

        # save the given file as a pydub.AudioSegment
        if file not in (".ipynb_checkpoints"):
            segment = AudioSegment.from_file(f"{dirpath}{file}")
        
        # repeat the length of the sound to atleast 30 seconds
        # (expressed in milliseconds)
            while len(segment) < 30000:
                segment = segment * 2
        
            # add the segment to the playlist
            playlist.append(segment)
            
            # stop if the length of our playlist is equal
            # to the number of desired samples
            if n_samples >= 1 and len(playlist) >= n_samples:
                return playlist
    
    return playlist


def save_new_audio(playlist:list[AudioSegment], dirpath:str,  mammal:str, chunk_len:int=30000) -> None:
    '''Make 30 second chuncks for each wav file in a given playlist.

    Params:
        playlist [list[AudioSegment]]: a list of extended raw audio
            segments 
        dirpath [str]: the path where the new audio chunks will be
            saved. This is typically the training data folder.
        mammal [str]: the name of the mammal associated with the 
            audio file. 
    Raises:
        None
    Return:
        None
    '''
    
    # make sure we have a slash at the end of the directory
    if dirpath[0] != '/':
       dirpath += '/'

    for idx, wav in enumerate(playlist):
        audio = make_chunks(wav, chunk_len)
        audio[0].export(f"{dirpath}/{mammal}_{idx}.wav", format="wav")


def get_wavs_from_training(src_path:str, dst_path:str, mammal:str) -> None:
    '''Augment the audio
    
    Params:
        src_path [str]: the path where audio chunks are stored. These
            will be augmented and placed in the dst_path.
        dst_path [str]: the path where the new audio chunks will be
            saved. This is the augmented data folder and will be used
            for training.
        mammal [str]: the name of the mammal associated with the 
            audio file. 
    Raises:
        None
    Returns:
        None
    '''
    
    # create an empty playlist to hold the augemented audio files
    aug_playlist = []
    
    # walk through all files in the extended audio file directory
    for file in os.listdir(src_path):
        try:
            if file not in (".ipynb_checkpoints"):
                sound = AudioSegment.from_file(f"{src_path}{file}")
                aug_playlist.append(sound)
        except IsADirectoryError as e:
            continue
            
    for idx, sound in enumerate(aug_playlist):

        # randomly choose +/- 2 semitones
        octaves = random.choice([0.1667, -0.1667])

        # randomly choose +/- 3 dB
        dB = random.choice([3, -3])
        
        new_sample_rate = int(sound.frame_rate * (2.0 ** octaves))
        new_sound = sound._spawn(sound.raw_data, overrides={'frame_rate': new_sample_rate})
        
        # grab the first 15 and last 15 and concatenate
        seconds = 15 * 1000
            
        first = new_sound[:seconds]
        last = new_sound[-seconds+1:] # add the - to get the last 15 sec
            
        aug_samp = first + last
        aug_samp = aug_samp + dB # increase/decrease volume 
        
        # export augmented audio into its appropriate class
        aug_samp.export(f"{dst_path}{mammal}_aug_{idx}.wav", format="wav")