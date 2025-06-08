import gradio as gr
import os

import modnet

root_path = os.environ.get("ROOT_PATH")


def modnet_photos_transfer(image, crop=False):

    return modnet.predict(image, crop)

demo = gr.Interface(
    fn=modnet_photos_transfer, 
    inputs=[
        "image",
        gr.Checkbox(label="是否进行裁剪", value=False)
    ],
    outputs="image")
demo.launch(server_name="0.0.0.0", root_path=root_path)