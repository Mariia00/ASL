# file: gui_launcher.py

import gradio as gr
from sign_language_recognition7 import process_image
import cv2


with gr.Blocks(title="Graphical GUI for Sign Language Recognition") as demo:
    gr.Markdown("# 🖐️ Sign Language to Gemini Chatbot")
    gr.Markdown("Upload or stream your sign language video, and let the model decode it.")

    with gr.Row():
        webcam = gr.Image(source="webcam", streaming=True, label="Live Feed")
        output_img = gr.Image(label="Processed Frame")

    with gr.Row():
        output_sentence = gr.Textbox(label="Recognized Sentence")
        output_status = gr.Textbox(label="Status Message")

    def run_inference(image):
        frame, sentence, status = process_image(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame, sentence, status

    webcam.change(fn=run_inference, inputs=webcam, outputs=[output_img, output_sentence, output_status])

if __name__ == '__main__':
    demo.launch()
