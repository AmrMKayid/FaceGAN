import os
import sys
from pathlib import Path

import cv2


class MultiFaceCropper:
  CASCADE_PATH = "./data/face_cascade.xml"
  COUNT = 1

  def __init__(
      self,
      cropped_size=500,
      radius=100,
  ) -> None:
    self.face_cascade = cv2.CascadeClassifier(MultiFaceCropper.CASCADE_PATH)
    self.cropped_size = 500
    self.radius = 100

  def generate(
      self,
      image_path: str = '',
      show_result=False,
  ) -> None:

    image_path = Path(image_path)
    img_name = image_path.name.split('.')[0]

    img = cv2.imread(str(image_path))

    if (img is None):
      print("Can't open image file")
      return 0

    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = self.face_cascade.detectMultiScale(
        img,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(250, 250),
    )

    if (faces is None):
      print('Failed to detect face')
      return 0

    if (show_result):
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


if __name__ == '__main__':
  args = sys.argv
  argc = len(args)

  if (argc != 2):
    print('Usage: %s [image file]' % args[0])
    quit()

  detecter = MultiFaceCropper()
  detecter.generate(args[1])
