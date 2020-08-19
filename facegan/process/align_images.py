import os
from glob import glob
from pathlib import Path
from typing import Union, List

import numpy as np
from tensorflow.keras.utils import get_file

from facegan import ROOT_PATH
from facegan.ffhq.face_alignment import image_align
from facegan.ffhq.landmarks_detector import LandmarksDetector
from facegan.utils.utils import unpack_bz2


class FaceAligner:
  LANDMARKS_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'

  def __init__(self):
    self.landmarks_model_path = unpack_bz2(
        get_file(
            'shape_predictor_68_face_landmarks.dat.bz2',
            FaceAligner.LANDMARKS_MODEL_URL,
            cache_subdir=f'{ROOT_PATH}/data/models',
        ))

    self.landmarks_detector = LandmarksDetector(self.landmarks_model_path)

  def align(
      self,
      img: Union[np.ndarray, str],
      img_name: str = 'image.png',
      aligned_dir=f'{ROOT_PATH}/data/aligned',
      output_size=1024,  # The dimension of images for input to the model
      x_scale=1,  # Scaling factor for x dimension
      y_scale=1,  # Scaling factor for y dimension
      em_scale=0.1,  # Scaling factor for eye-mouth distance
      use_alpha=False,  # Add an alpha channel for masking
  ) -> np.ndarray:

    print('Aligning %s ...' % img_name)
    aligned_img = None
    for i, face_landmarks in enumerate(
        self.landmarks_detector.get_landmarks(img),
        start=1,
    ):
      try:
        print('Starting face alignment...')
        aligned_face_path = os.path.join(aligned_dir, img_name)  # TODO: mkdir
        aligned_img = image_align(
            img,
            aligned_face_path,
            face_landmarks,
            output_size=output_size,
            x_scale=x_scale,
            y_scale=y_scale,
            em_scale=em_scale,
            alpha=use_alpha,
        )
        print('Wrote result %s' % aligned_face_path)
      except Exception as e:
        print(f'Exception in face alignment! -> {e}')

    return aligned_img

  def auto_align(
      self,
      raw_dir=f'{ROOT_PATH}/data/cropped',  # Directory with raw images for face alignment
      aligned_dir=f'{ROOT_PATH}/data/aligned',  # Directory for storing aligned images
  ) -> List[np.ndarray]:

    imgs = []
    images = glob(f'{raw_dir}/*.png') + glob(f'{raw_dir}/*.jpg')
    for img_name in images:
      raw_img_path = img_name  # os.path.join(raw_dir, img_name)
      face_img_name = f'{Path(raw_img_path).name.split(".")[0]}.png'
      img = self.align(
          img=raw_img_path,
          img_name=face_img_name,
          aligned_dir=aligned_dir,
      )
      imgs.append(img)
    return imgs
