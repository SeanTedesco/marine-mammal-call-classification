import gradio as gr
import os
import librosa
import math
import json

mammal_call_path = r'C:\Users\seant\Documents\marine-mammal-call-classification\data_augmented\KillerWhale\KillerWhale_aug_0.wav'
# mammal_call_path = '/Users/seantedesco/Documents/marine-mammal-call-classification/data_augmented/KillerWhale/KillerWhale_aug_0.wav'
json_path = '/Users/seantedesco/Documents/marine-mammal-call-classification/mfccs_gui_test.json'

def select_model_given_constraint(desired_size):
    pass

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

def save_mfcc_single(audio_file, mammal_label, mammal_name, export_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
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
    mammal_image_path = r"C:\Users\seant\Documents\marine-mammal-call-classification\images\gui-images\killer-whale.jpg"
    #mammal_image_path = '/Users/seantedesco/Documents/marine-mammal-call-classification/images/gui-images/killer-whale.jpg'
    
    mammal_label, mammal_name = label_and_name_from_audio_file(audio_file)
    save_mfcc_single(audio_file, mammal_label, mammal_name, json_path)
    chosen_model = select_model_given_constraint(desired_size)
    return mammal_image_path, f"label for {audio_file}"

with gr.Blocks() as demo:
    gr.Markdown("# Marine Mammal Classification")

    with gr.Box():
        gr.Markdown("## Listen")
        file_path_input = gr.Textbox(mammal_call_path)
        audio = gr.Audio()

    with gr.Box():
        gr.Markdown('## Model Constraints')
        model_size_slider = gr.Slider(0, 16, 4, step=0.25)

    with gr.Box():
        gr.Markdown("## Classify")
        with gr.Row():
            classify_button = gr.Button("Classify")
            class_label_output = gr.Textbox()
        image_output = gr.Image()
        classify_button.click(classify, inputs=[audio, file_path_input, model_size_slider], outputs=[image_output, class_label_output])

demo.launch()