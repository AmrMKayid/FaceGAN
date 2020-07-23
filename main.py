import gradio as gr
import numpy as np

from align_images import align

gr.Interface(
    align,
    gr.inputs.Image(shape=(500, 500)),
    "image",
).launch(share=True)
