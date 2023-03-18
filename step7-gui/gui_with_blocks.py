import gradio as gr

mammal_call_path = r'C:\Users\seant\Documents\marine-mammal-call-classification\data_augmented\KillerWhale\KillerWhale_aug_0.wav'


def classify(audio_file, desired_size):
    mammal_image_path = r"C:\Users\seant\Documents\marine-mammal-call-classification\images\gui-images\killer-whale.jpg"
    
    print(f"desired_mem: {desired_size}")
    return mammal_image_path, f"label for {audio_file}"

with gr.Blocks() as demo:
    gr.Markdown("# Marine Mammal Classification")

    with gr.Box():
        gr.Markdown("## Listen")
        file = gr.File(mammal_call_path)
        audio = gr.Audio(mammal_call_path)

    with gr.Box():
        gr.Markdown('## Model Constraints')
        model_size_slider = gr.Slider(0, 16, 4, step=0.25)

    with gr.Box():
        gr.Markdown("## Classify")
        with gr.Row():
            classify_button = gr.Button("Classify")
            class_label_output = gr.Textbox()
        image_output = gr.Image()
        classify_button.click(classify, inputs=[file, model_size_slider], outputs=[image_output, class_label_output])

demo.launch()