import gradio as gr
import os
import librosa
import math
import json
import tensorflow as tf
import numpy as np
from compression_lib import load_cnn_json

mammal_call_path    = '/Users/seantedesco/Documents/marine-mammal-call-classification/data_augmented/KillerWhale/KillerWhale_aug_0.wav'
export_path         = '/Users/seantedesco/Documents/marine-mammal-call-classification/mfccs_gui_test.json'
saved_model_path    = '/Users/seantedesco/Documents/marine-mammal-call-classification/saved_model/layers3/trial1/'

def select_image_for_classification(mammal_label:int):
    image_map = {
        1: '/Users/seantedesco/Documents/marine-mammal-call-classification/images/gui-images/bowhead-whale.jpeg',
        2: '/Users/seantedesco/Documents/marine-mammal-call-classification/images/gui-images/humpback-whale.jpeg',
        3: '/Users/seantedesco/Documents/marine-mammal-call-classification/images/gui-images/killer-whale.jpg',
        4: '/Users/seantedesco/Documents/marine-mammal-call-classification/images/gui-images/walrus.jpg',
        5: '/Users/seantedesco/Documents/marine-mammal-call-classification/images/gui-images/fin-back-whale.jpg',
        6: '/Users/seantedesco/Documents/marine-mammal-call-classification/images/over-under-fitting-layers-3-5-7.png'
    }
    return image_map[mammal_label]

def format_predictions(predictions):
    return f'''
            BowheadWhale: {predictions[1]:.5f}
            HumpbackWhale: {predictions[2]:.5f}
            KillerWhale: {predictions[3]:.5f}
            Walrus: {predictions[4]:.5f}
            FinBackWhale: {predictions[5]:.5f}
            EmptyOcean: {predictions[6]:.5f}
            '''

def select_model_given_constraint(desired_size):
    size_to_model_index_map = {
        # Model Size in MB : Model Index in List
    }

    model_list = []
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(saved_model_path)):
        for f in filenames:
            loaded_model = tf.keras.models.load_model(dirpath+f)
            model_list.append(loaded_model)
    return model_list[0]

def label_and_name_from_audio_file(audio_file):
    label_map = {
        "BowheadWhale": 1,
        "HumpbackWhale": 2,
        "KillerWhale": 3,
        "Walrus": 4,
        "Fin_FinbackWhale": 5,
        "EmptyOcean": 6
    }
    file_segments = audio_file.split('/')
    mammal_name = file_segments[-2]
    return label_map[mammal_name], mammal_name

def save_mfcc_single(audio_file, mammal_label, mammal_name, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    SAMPLE_RATE = 22050
    DURATION = 30 # measured in seconds
    SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

    # dictionary to store data
    data = {
        "mapping": [],
        "mfcc": [],
        "labels": [],
    }

    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length) # 1.2 -> 2

    signal, sr = librosa.load(audio_file, sr=SAMPLE_RATE)

    # process segments extracting mfcc and storing the data
    for s in range(num_segments):
        start_sample = num_samples_per_segment * s # s =0 -> 0
        finish_sample = start_sample + num_samples_per_segment # s=0 -> num_samples_per_segment

        mfcc = librosa.feature.mfcc(y=signal[start_sample:finish_sample],
                                    sr=sr,
                                    n_fft=n_fft,
                                    n_mfcc=n_mfcc,
                                    hop_length=hop_length)
        mfcc = mfcc.T

        # store mfcc for segment if it has the expected length
        if len(mfcc) == expected_num_mfcc_vectors_per_segment: #  and semantic_label not in ('.ipynb_checkpoints')
            data['mfcc'].append(mfcc.tolist())
            data['labels'].append(mammal_label)
            data['mapping'].append(mammal_name)

    with open(export_path, "w") as fp:
        json.dump(data, fp, indent=4)

def classify(raw_audio, audio_file, desired_size):
    
    # prepare the audio data
    mammal_label, mammal_name = label_and_name_from_audio_file(audio_file)
    save_mfcc_single(audio_file, mammal_label, mammal_name)
    X, y, L = load_cnn_json(export_path)

    # prepare and predict with a cnn model
    chosen_model = select_model_given_constraint(desired_size)
    predictions = chosen_model.predict(X)[0]

    # format GUI outputs
    prediction_string = format_predictions(predictions)
    prediction_idx = np.argmax(predictions)
    mammal_image = select_image_for_classification(prediction_idx)

    return mammal_image, prediction_string

with gr.Blocks() as demo:
    gr.Markdown("# Marine Mammal Classification")

    with gr.Box():
        gr.Markdown("## Listen")
        file_path_input = gr.Textbox(mammal_call_path)
        audio = gr.Audio(mammal_call_path)

    with gr.Box():
        gr.Markdown('## Model Constraints')
        model_size_slider = gr.Slider(0, 12, 6, step=0.25)

    with gr.Box():
        gr.Markdown("## Classify")
        with gr.Row():
            classify_button = gr.Button("Classify")
            class_label_output = gr.Textbox()
        image_output = gr.Image()
        classify_button.click(classify, inputs=[audio, file_path_input, model_size_slider], outputs=[image_output, class_label_output])

demo.launch()