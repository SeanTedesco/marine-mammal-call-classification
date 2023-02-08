import gradio as gr

def predict(audio, num_params, layers):
    return "Demo"

mammal_call_path = '/Users/emmap/Documents/marine-mammal-call-classification/data_augmented/KillerWhale/KillerWhale_aug_0.wav'

demo = gr.Interface(
    fn=predict, 
    inputs=[gr.Audio(value=mammal_call_path), gr.Slider(0, 100), gr.Radio(["add", "subtract", "multiply", "divide"])],
    outputs="text",
    title= "MMM Tasty",
    description="We are here to classify them whale sounds!",
    article="Check our [marine-mammal-call-classification](https://github.com/SeanTedesco/marine-mammal-call-classification/blob/master/step7-gui/gui_launcher.py) repository GitHub Page" 
)

demo.launch(share=True) 