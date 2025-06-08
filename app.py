import gradio as gr
import os

import modnet

root_path = os.environ.get("ROOT_PATH")


def modnet_photos_transfer(image):

    return modnet.predict(image)


demo = gr.Interface(fn=modnet_photos_transfer, inputs="image", outputs="image")
demo.launch(server_name="0.0.0.0", root_path=root_path)