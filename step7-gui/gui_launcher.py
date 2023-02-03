import gradio as gr

def predict(audio):
    return "Demo"


mammal_call_path = '/Users/seantedesco/Documents/marine-mammal-call-classification/data_augmented/KillerWhale/KillerWhale_aug_0.wav'

demo = gr.Interface(fn=predict, inputs=gr.Audio(value=mammal_call_path), outputs="text")

demo.launch(share=True) 