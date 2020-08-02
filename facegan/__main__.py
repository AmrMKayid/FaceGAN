# import gradio as gr
# import numpy as np
#
# from align_images import align
#
# gr.Interface(
#     align,
#     gr.inputs.Image(shape=(500, 500)),
#     "image",
# ).launch(share=False)

import cv2
import numpy as np
from PIL import Image

from facegan.process.multifaces import MultiFaceCropper

if __name__ == '__main__':
    test_path = './data/raw/6faces.jpg'
    mfc = MultiFaceCropper()
    img = cv2.imread(test_path)
    # print(img)
    images = mfc.crop(img)
    print(images, len(images), images[0].shape)
