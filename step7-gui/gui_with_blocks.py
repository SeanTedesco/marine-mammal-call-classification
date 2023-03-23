import gradio as gr
import os
import librosa
import math
import json
import tensorflow as tf
import numpy as np
from compression_lib import load_cnn_json
import random

export_path         = 'mfccs_gui_test.json'
# saved_model_path    = '../saved_model/layers3/'

MODEL_PATH_PREFIX = '../saved_model/gui_models/'
# IMPORTANT: index of size and model name must match!
MODEL_SIZE = ["0.15", "1.1", "2.1", "3.3"]
FILE_NAMES = ["infineon_aurix.h5", "infineon_aurix.h5", "sifive.h5", "sifive.h5"]

def select_image_for_classification(mammal_label:int):
    image_map = {
        1: '../images/gui-images/bowhead-whale.jpeg',
        2: '../images/gui-images/humpback-whale.jpeg',
        3: '../images/gui-images/killer-whale.jpg',
        4: '../images/gui-images/walrus.jpg',
        5: '../images/gui-images/fin-back-whale.jpg',
        6: '../images/gui-images/empty_ocean.avif'
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
    index = MODEL_SIZE.index(desired_size)
    file_name = MODEL_PATH_PREFIX + FILE_NAMES[index]
    loaded_model = tf.keras.models.load_model(file_name)
    return loaded_model

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

    if desired_size == 0.5:
        desired_size = MODEL_SIZE[0]
    elif desired_size == 1 or desired_size == 1.5:
        desired_size = MODEL_SIZE[1]
    elif desired_size == 2 or desired_size == 2.5:
        desired_size = MODEL_SIZE[2]
    else:
        desired_size = MODEL_SIZE[3]

    # prepare and predict with a cnn model
    chosen_model = select_model_given_constraint(desired_size)
    predictions = chosen_model.predict(X)[0]

    # format GUI outputs
    prediction_string = format_predictions(predictions)
    prediction_idx = np.argmax(predictions)
    mammal_image = select_image_for_classification(prediction_idx)

    return mammal_image, prediction_string


def get_name(input):
    # Convert to the species for filename.
    if input == "Killer Whale":
        input = "KillerWhale"
    if input == "Bowhead Whale":
        input = "BowheadWhale"
    if input == "Humpback Whale":
        input = "HumpbackWhale"
    if input == "Finback Whale":
        input = "Fin_FinbackWhale"
    if input == "Ambient Ocean Noise":
        input = "EmptyOcean"
    
    # Randomly generate number
    file_number = random.randint(0, 99)

    audio_file_path = "../data_augmented/" + input + "/" + input + "_aug_" + str(file_number) + ".wav"
    return audio_file_path, audio_file_path


def get_microcontroller_image(input):
    if input == 0.5:
        return "../images/gui-images/microcontrollers/arduino_mega.jpeg", "Arduino Mega"
    elif input == 1 or input == 1.5 or input == 2:
        return "../images/gui-images/microcontrollers/stm32.webp", "STM32"
    elif input == 2.5 or input == 3 or input == 3.5:
        return "../images/gui-images/microcontrollers/infineon.png", "Infineon"


with gr.Blocks(theme = gr.themes.Soft(primary_hue="sky")) as demo:
    gr.Markdown("# Marine Mammal Classification")

    with gr.Box(css=".gradio-container {background-color: sky}"):
        gr.Markdown("## Listen")
        species_name = gr.Dropdown(
            ["Walrus", "Killer Whale", "Finback Whale", "Bowhead Whale", "Humpback Whale", "Ambient Ocean Noise"], label="Species", info="Please select a species to classify:"
        )

        file_path_input = gr.Textbox(label="Audio file path")
        audio = gr.Audio()
        species_name.change(fn = get_name, inputs = species_name, outputs = [file_path_input, audio])

    with gr.Box():
        gr.Markdown('## Model Constraints')
        model_size_slider = gr.Slider(0.5, 3.5, 1, step=0.5, label="Slide to choose your size in MB.")
        microcontroller_image = gr.Image()
        microcontroller_name = gr.Textbox(label="Microcontroller brand name:")
        microcontroller_image.style(height=200, width=200)
        model_size_slider.change(fn = get_microcontroller_image, inputs = model_size_slider, outputs = [microcontroller_image, microcontroller_name])

    with gr.Box():
        gr.Markdown("## Classification")
        with gr.Row():
            classify_button = gr.Button("Click here to classify!")
            classify_button.style(size="lg")
        with gr.Row():
            class_label_output = gr.Textbox(label="Result:", show_label=True)
        image_output = gr.Image()
        classify_button.click(classify, inputs=[audio, file_path_input, model_size_slider], outputs=[image_output, class_label_output])

demo.launch(share=True) # share=True gives a public link.