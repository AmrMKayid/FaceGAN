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

# import cv2

from facegan.faces.styleganencoder import StyleGANEncoder
# from facegan.process.align_images import FaceAligner
# from facegan.process.multifaces import MultiFaceCropper

if __name__ == '__main__':
  # test_path = './data/raw/test.jpg'
  # mfc = MultiFaceCropper()
  # img = cv2.imread(test_path)
  # # print(img)
  # images = mfc.crop(img)
  # print(images, len(images), images[0].shape)
  #
  # face_aligner = FaceAligner()
  # face_aligner.auto_align()

  sg_encoder = StyleGANEncoder()
  print(vars(sg_encoder))
