from glob import glob
from pathlib import Path

import cv2
import numpy as np
from typing import List


class MultiFaceCropper:
  CASCADE_PATH = "./data/face_cascade.xml"
  COUNT = 1

  def __init__(
      self,
      data_dir='./data/raw',
      cropped_size=1024,
      radius=500,
  ) -> None:
    self.face_cascade = cv2.CascadeClassifier(MultiFaceCropper.CASCADE_PATH)
    self.data_dir = data_dir
    self.cropped_size = cropped_size
    self.radius = radius

  def crop(
      self,
      image: np.ndarray = None,
  ) -> List[np.ndarray]:
    return self._crop(image)

  def auto_crop(self) -> List[np.ndarray]:

    images = glob(f'{self.data_dir}/*.png') + glob(f'{self.data_dir}/*.jpg')
    for image_path in images:
      image_path = Path(image_path)
      print(f'Detecting faces in {image_path}')
      img_name = image_path.name.split('.')[0]

      img = cv2.imread(str(image_path))

      _ = self._crop(img, img_name)

  def _crop(
      self,
      img,
      img_name: str = 'cropped_image',
      show_result: bool = False,
  ) -> List[np.ndarray]:
    images = []

    if (img is None):
      print("Can't open image file")
      return 0

    faces = self.face_cascade.detectMultiScale(
        img,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(250, 250),
    )

    if faces is None:
      print('Failed to detect face')
      return 0

    if show_result:
      for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
      cv2.imshow('img', img)
      cv2.waitKey(0)
      cv2.destroyAllWindows()

    faces_count = len(faces)
    print("Detected faces: %d" % faces_count)
    height, width = img.shape[:2]

    for (x, y, w, h) in faces:
      face_img = img[y - self.radius:y + h + self.radius,
                     x - self.radius:x + w + self.radius]
      last_img = cv2.resize(
          face_img,
          (self.cropped_size, self.cropped_size),
      )
      cv2.imwrite(
          f"./data/cropped/{img_name}_{MultiFaceCropper.COUNT}.png",
          last_img,
      )
      MultiFaceCropper.COUNT += 1

      images.append(last_img)

    return images
